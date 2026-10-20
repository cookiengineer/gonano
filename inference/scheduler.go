package inference

import (
	"sort"
	"strings"

	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tensors"
	"github.com/cookiengineer/gonano/tokenizer"
)

// SequenceStatus is the lifecycle state of a scheduled request.
type SequenceStatus uint8

const (
	SequenceWaiting SequenceStatus = iota
	SequencePrefill
	SequenceDecode
	SequenceFinished
	SequenceFailed
)

// PreemptionPolicy selects how the scheduler makes room for waiting requests.
type PreemptionPolicy uint8

const (
	// PreemptNone only queues waiting requests.
	PreemptNone PreemptionPolicy = iota
	// PreemptLongest releases the running sequence with the largest reservation
	// and recomputes it later from its prompt plus generated tokens.
	PreemptLongest
)

// Request describes one scheduler request.
type Request struct {
	ID          string
	Prompt      []int
	MaxTokens   int
	Temperature float32
	TopK        int
	TopP        float32
	MinP        float32
	Seed        uint64
	Stop        []string
	Options     GenerateOptions
	Constraint  Constraint
	// Weights optionally overrides the ensemble member weights for this
	// request. The single-model scheduler ignores it.
	Weights []float32
}

// Sequence is one in-flight request.
type Sequence struct {
	ID           string
	Prompt       []int
	Tokens       []int
	Generated    []int
	Masks        []int
	Cache        *model.KVBuffer
	Status       SequenceStatus
	FinishReason string
	Prefilled    int
	Position     int
	// Row is the sequence's row in Scheduler.workCache; -1 for solo requests.
	Row int
	// Solo marks a request that cannot join the mixed work cache (CED or
	// speculative decoding) and owns a private cache.
	Solo bool
	// Logprobs holds one entry per sampled token when requested.
	Logprobs []TokenLogprob

	reservedBlocks int
	preempted      int
	weights        []float32
	forcedTokens   []int
	inToolCall     bool
	toolCallTokens []int
	inThinking     bool
	thinkingTokens int

	request    Request
	rng        *tensors.RNG
	lastLogits []float32
	content    strings.Builder
	constraint Constraint
}

// Event is emitted when a scheduler step produces a token or finishes a
// sequence.
type Event struct {
	Sequence     *Sequence
	Token        int
	TokenMask    int
	Finished     bool
	FinishReason string
	// Logprob is set for sampled tokens when logprobs were requested.
	Logprob *TokenLogprob
}

// SchedulerStats reports scheduler progress.
type SchedulerStats struct {
	Waiting       int
	Running       int
	Finished      int
	Generated     int
	PrefillTokens int64
	DecodeTokens  int64
	KVBlocksInUse int
	KVBlocksFree  int
	Preemptions   int
}

// Scheduler interleaves prefill and batches decode across independent
// sequences. Prefill remains per-sequence and chunked; every running decode row
// advances together in one model forward. All rows share one paged block
// manager, so a prefilled row joins the decode cache without copying its KV.
type Scheduler struct {
	Engine       *Engine
	MaxActive    int
	PrefillChunk int
	// Blocks optionally supplies the shared paged block pool. A nil pool is
	// created from MaxActive and the model sequence length.
	Blocks *model.KVBlockManager
	// Prefix, when non-nil, enables block-aligned prefix reuse. Snapshots are
	// copied into each request's cache; the cache itself is in-memory only.
	Prefix *BlockPrefixCache
	// Preemption selects whether admission may evict a running request.
	Preemption PreemptionPolicy

	waiting   []*Sequence
	running   []*Sequence
	finished  []*Sequence
	workCache *model.KVBuffer
	rows      []*Sequence
	pools     *model.KVBlockPools
	reserved  int
	nextSeed  uint64
	stats     SchedulerStats
}

// NewScheduler builds a scheduler. maxActive <= 0 means unbounded; the decode
// cache then defaults to eight rows.
func NewScheduler(engine *Engine, maxActive int) *Scheduler {
	return &Scheduler{Engine: engine, MaxActive: maxActive, nextSeed: 1}
}

// Submit queues a request. A zero seed is replaced by a deterministic scheduler
// seed.
func (scheduler *Scheduler) Submit(request Request) *Sequence {
	if request.MaxTokens <= 0 {
		request.MaxTokens = scheduler.Engine.Model.Config.SequenceLen
	}
	seed := request.Seed
	if seed == 0 {
		seed = scheduler.nextSeed
		scheduler.nextSeed++
	}
	constraint := request.Constraint
	if constraint == nil {
		constraint = request.Options.Constraint
	}
	sequence := &Sequence{
		ID:         request.ID,
		Prompt:     append([]int(nil), request.Prompt...),
		Tokens:     append([]int(nil), request.Prompt...),
		Status:     SequenceWaiting,
		Row:        -1,
		request:    request,
		rng:        tensors.NewRNG(seed),
		lastLogits: nil,
		constraint: cloneConstraint(constraint),
	}
	if (request.Options.Thinking || request.Options.ThinkingBudget > 0) && len(request.Prompt) > 0 {
		thinkStart := scheduler.Engine.Tokenizer.EncodeSpecial("<|think_start|>")
		if request.Prompt[len(request.Prompt)-1] == thinkStart {
			sequence.inThinking = true
		}
	}
	scheduler.waiting = append(scheduler.waiting, sequence)
	return sequence
}

// Cancel removes a sequence from the scheduler and releases its cache state.
// It must only be called from the goroutine that drives Step.
func (scheduler *Scheduler) Cancel(sequence *Sequence) {
	if sequence == nil {
		return
	}
	for index, candidate := range scheduler.waiting {
		if candidate == sequence {
			scheduler.waiting = append(scheduler.waiting[:index], scheduler.waiting[index+1:]...)
			sequence.Status = SequenceFailed
			return
		}
	}
	for index, candidate := range scheduler.running {
		if candidate != sequence {
			continue
		}
		if sequence.Cache == scheduler.workCache && sequence.Cache != nil {
			scheduler.workCache.ReleaseRow(sequence.Row)
			if sequence.Row >= 0 && sequence.Row < len(scheduler.rows) {
				scheduler.rows[sequence.Row] = nil
			}
		} else if sequence.Cache != nil {
			sequence.Cache.Release()
		}
		sequence.Cache = nil
		sequence.Row = -1
		sequence.Solo = false
		scheduler.reserved -= sequence.reservedBlocks
		if scheduler.reserved < 0 {
			scheduler.reserved = 0
		}
		sequence.reservedBlocks = 0
		sequence.Status = SequenceFailed
		scheduler.running = append(scheduler.running[:index], scheduler.running[index+1:]...)
		scheduler.compactRows()
		return
	}
}

// Pending reports whether the scheduler still has work.
func (scheduler *Scheduler) Pending() bool {
	return len(scheduler.waiting) > 0 || len(scheduler.running) > 0
}

// Stats returns a snapshot of scheduler counters.
func (scheduler *Scheduler) Stats() SchedulerStats {
	stats := scheduler.stats
	stats.Waiting = len(scheduler.waiting)
	stats.Running = len(scheduler.running)
	stats.Finished = len(scheduler.finished)
	stats.Preemptions = scheduler.stats.Preemptions
	if scheduler.Blocks != nil {
		stats.KVBlocksInUse = scheduler.Blocks.InUse()
		stats.KVBlocksFree = scheduler.Blocks.FreeBlocks()
	}
	return stats
}

// Run executes until every submitted request finishes.
func (scheduler *Scheduler) Run() []*Sequence {
	for len(scheduler.waiting) > 0 || len(scheduler.running) > 0 {
		scheduler.Step()
	}
	return scheduler.finished
}

// Step admits waiting requests, advances prefill and decode rows, and runs one
// mixed T=1 forward when both kinds are active. When only prefills are active,
// a chunked prefill forward is used instead.
func (scheduler *Scheduler) Step() []Event {
	scheduler.ensureDecodeCache()
	scheduler.admit()

	events := []Event{}
	sampled := make(map[*Sequence]bool)

	for _, sequence := range scheduler.running {
		if sequence.Status != SequencePrefill || !sequence.Solo {
			continue
		}
		if scheduler.speculativeEligible(sequence) {
			events = append(events, scheduler.runSpeculative(sequence)...)
		} else {
			scheduler.prefill(sequence)
		}
	}

	hasDecode := false
	for _, sequence := range scheduler.running {
		if !sequence.Solo && sequence.Status == SequenceDecode {
			hasDecode = true
			break
		}
	}
	if !hasDecode {
		prefillEvents, promoted := scheduler.chunkedPrefill()
		events = append(events, prefillEvents...)
		for sequence := range promoted {
			sampled[sequence] = true
		}
	}

	for _, sequence := range scheduler.running {
		if sequence.Status != SequenceDecode || sampled[sequence] {
			continue
		}
		events = append(events, scheduler.emit(sequence)...)
	}

	scheduler.reap()
	scheduler.compactRows()

	runMixed := hasDecode || len(sampled) > 0
	if runMixed {
		scheduler.mixedForward()
	}
	scheduler.forwardSolo()
	scheduler.reap()
	scheduler.compactRows()

	for _, event := range events {
		if event.TokenMask == 1 {
			scheduler.stats.Generated++
		}
	}
	return events
}

// chunkedPrefill advances every non-solo prefill row by one common chunk. It is
// only used when no work-cache decode rows are active.
func (scheduler *Scheduler) chunkedPrefill() ([]Event, map[*Sequence]bool) {
	events := []Event{}
	promoted := make(map[*Sequence]bool)
	scheduler.compactRows()

	active := make([]*Sequence, 0, len(scheduler.rows))
	for _, sequence := range scheduler.rows {
		if sequence != nil && sequence.Status == SequencePrefill {
			active = append(active, sequence)
		}
	}
	if len(active) == 0 {
		return events, promoted
	}
	config := scheduler.Engine.Model.Config
	chunk := 0
	for _, sequence := range active {
		remaining := len(sequence.Tokens) - sequence.Prefilled
		if remaining <= 0 {
			continue
		}
		if chunk == 0 || remaining < chunk {
			chunk = remaining
		}
	}
	if chunk <= 0 {
		return events, promoted
	}
	if scheduler.PrefillChunk > 0 && chunk > scheduler.PrefillChunk {
		chunk = scheduler.PrefillChunk
	}
	if scheduler.Prefix != nil {
		for _, sequence := range active {
			position := scheduler.workCache.PositionAt(sequence.Row)
			distance := model.KVBlockSize - position%model.KVBlockSize
			if distance < chunk {
				chunk = distance
			}
		}
	}
	if chunk < 1 {
		chunk = 1
	}

	data := make([]int32, len(active)*chunk)
	for index, sequence := range active {
		copy(data[index*chunk:(index+1)*chunk], toI32(sequence.Tokens[sequence.Prefilled:sequence.Prefilled+chunk]))
	}
	scheduler.stats.PrefillTokens += int64(len(active) * chunk)
	ids := tensors.NewInt32sWithData([]int{len(active), chunk}, data)
	logits := scheduler.Engine.Model.Forward(ids, scheduler.workCache)
	vocab := config.VocabSize
	for index, sequence := range active {
		sequence.Prefilled += chunk
		sequence.lastLogits = logitsRow(logits, index*chunk+chunk-1, vocab)
		position := scheduler.workCache.PositionAt(sequence.Row)
		if scheduler.Prefix != nil && position > 0 && position%model.KVBlockSize == 0 {
			if snapshot := scheduler.workCache.CloneRow(sequence.Row); snapshot != nil {
				scheduler.Prefix.Store(sequence.Tokens[:position], snapshot)
				snapshot.Release()
			}
		}
		if sequence.Prefilled == len(sequence.Tokens) {
			sequence.Status = SequenceDecode
			events = append(events, scheduler.emit(sequence)...)
			promoted[sequence] = true
		}
	}
	return events, promoted
}

// mixedForward advances every work-cache row by one token in a single forward:
// decode rows feed their sampled token, prefill rows feed their next prompt
// token.
func (scheduler *Scheduler) mixedForward() {
	active := make([]*Sequence, 0, len(scheduler.rows))
	for _, sequence := range scheduler.rows {
		if sequence != nil {
			active = append(active, sequence)
		}
	}
	if len(active) == 0 {
		return
	}
	tokens := make([]int32, len(active))
	positions := make([]*Sequence, len(active))
	count := 0
	for _, sequence := range active {
		switch sequence.Status {
		case SequenceDecode:
			tokens[count] = int32(sequence.Tokens[len(sequence.Tokens)-1])
			scheduler.stats.DecodeTokens++
		case SequencePrefill:
			if sequence.Prefilled >= len(sequence.Tokens) {
				sequence.Status = SequenceDecode
				continue
			}
			tokens[count] = int32(sequence.Tokens[sequence.Prefilled])
			scheduler.stats.PrefillTokens++
		default:
			continue
		}
		positions[count] = sequence
		count++
	}
	if count == 0 {
		return
	}
	ids := tensors.NewInt32sWithData([]int{count, 1}, tokens[:count])
	logits := scheduler.Engine.Model.Forward(ids, scheduler.workCache)
	vocab := scheduler.Engine.Model.Config.VocabSize
	for index := 0; index < count; index++ {
		sequence := positions[index]
		sequence.lastLogits = logitsRow(logits, index, vocab)
		if sequence.Status != SequencePrefill {
			continue
		}
		sequence.Prefilled++
		position := scheduler.workCache.PositionAt(sequence.Row)
		if scheduler.Prefix != nil && position > 0 && position%model.KVBlockSize == 0 {
			if snapshot := scheduler.workCache.CloneRow(sequence.Row); snapshot != nil {
				scheduler.Prefix.Store(sequence.Tokens[:position], snapshot)
				snapshot.Release()
			}
		}
		if sequence.Prefilled == len(sequence.Tokens) {
			sequence.Status = SequenceDecode
		}
	}
}

// forwardSolo advances solo CED rows after they have been sampled.
func (scheduler *Scheduler) forwardSolo() {
	config := scheduler.Engine.Model.Config
	for _, sequence := range scheduler.running {
		if !sequence.Solo || sequence.Status != SequenceDecode || len(sequence.Tokens) == 0 {
			continue
		}
		token := sequence.Tokens[len(sequence.Tokens)-1]
		ids := tensors.NewInt32sWithData([]int{1, 1}, []int32{int32(token)})
		logits := scheduler.Engine.Model.Forward(ids, sequence.Cache)
		sequence.lastLogits = lastLogits(logits, config.VocabSize)
	}
}

func (scheduler *Scheduler) ensureDecodeCache() {
	if scheduler.workCache != nil {
		return
	}
	maxRows := scheduler.MaxActive
	if maxRows <= 0 {
		maxRows = 8
		scheduler.MaxActive = maxRows
	}
	config := scheduler.Engine.Model.Config
	if scheduler.Blocks == nil {
		blocksPerRow := (config.SequenceLen + model.KVBlockSize - 1) / model.KVBlockSize
		if blocksPerRow < 1 {
			blocksPerRow = 1
		}
		extra := 0
		if scheduler.Prefix != nil {
			entries := scheduler.Prefix.MaxEntries()
			if entries <= 0 {
				entries = maxRows
			}
			extra = entries * blocksPerRow
		}
		scheduler.Blocks = model.NewKVBlockManager(maxRows*(blocksPerRow+8) + extra)
	}
	scheduler.pools = model.NewKVBlockPools(scheduler.Blocks, config.NumLayer, config.NumKVHead, config.HeadDim())
	scheduler.workCache = newCacheForModelWithPools(scheduler.Blocks, scheduler.pools, config, maxRows, config.SequenceLen)
	scheduler.rows = make([]*Sequence, maxRows)
}

func (scheduler *Scheduler) admit() {
	limit := scheduler.MaxActive
	for len(scheduler.waiting) > 0 && (limit <= 0 || len(scheduler.running) < limit) {
		sequence := scheduler.waiting[0]
		scheduler.waiting = scheduler.waiting[1:]
		config := scheduler.Engine.Model.Config
		capacity := len(sequence.Prompt) + sequence.request.MaxTokens
		if capacity > config.SequenceLen {
			capacity = config.SequenceLen
		}
		if capacity < len(sequence.Prompt) {
			capacity = len(sequence.Prompt)
		}

		if config.CEDEnabled() || scheduler.speculativeEligible(sequence) {
			needed := scheduler.blocksFor(capacity)
			if !scheduler.ensureBudget(sequence, needed) {
				scheduler.waiting = append([]*Sequence{sequence}, scheduler.waiting...)
				break
			}
			scheduler.reserved += needed
			sequence.reservedBlocks = needed
			sequence.Cache = newCacheForModelWithPools(scheduler.Blocks, scheduler.pools, config, 1, capacity)
			sequence.Solo = true
			sequence.Prefilled = 0
			sequence.Status = SequencePrefill
			scheduler.running = append(scheduler.running, sequence)
			continue
		}

		row := scheduler.freeRow()
		if row < 0 {
			scheduler.waiting = append([]*Sequence{sequence}, scheduler.waiting...)
			break
		}
		needed := scheduler.blocksFor(capacity)
		var snapshot *model.KVBuffer
		matched := 0
		if scheduler.Prefix != nil {
			if cached, hit := scheduler.Prefix.Lookup(sequence.Tokens); cached != nil {
				if cached.BlockManager() == scheduler.Blocks {
					snapshot = cached
					matched = hit
					needed = scheduler.blocksFor(capacity - matched)
				} else {
					cached.Release()
				}
			}
		}
		if !scheduler.ensureBudget(sequence, needed) {
			if snapshot != nil {
				snapshot.Release()
			}
			scheduler.waiting = append([]*Sequence{sequence}, scheduler.waiting...)
			break
		}
		scheduler.reserved += needed
		sequence.reservedBlocks = needed
		sequence.Cache = scheduler.workCache
		sequence.Row = row
		scheduler.rows[row] = sequence
		scheduler.workCache.SetPositionAt(row, 0)
		if snapshot != nil {
			if scheduler.workCache.AdoptRow(row, snapshot) {
				sequence.Prefilled = matched
			} else {
				additional := scheduler.blocksFor(capacity) - needed
				scheduler.reserved += additional
				sequence.reservedBlocks += additional
			}
			snapshot.Release()
		}
		sequence.Status = SequencePrefill
		scheduler.running = append(scheduler.running, sequence)
	}
}

func (scheduler *Scheduler) ensureBudget(sequence *Sequence, needed int) bool {
	for {
		held := 0
		if scheduler.Prefix != nil {
			held = scheduler.Prefix.HeldBlocks()
		}
		if scheduler.reserved+held+needed <= scheduler.Blocks.BlockCount() || len(scheduler.running) == 0 {
			return true
		}
		if scheduler.Preemption == PreemptNone || sequence.preempted > 0 || scheduler.hasPreemptedWaiting() {
			return false
		}
		victim := scheduler.preemptionVictim()
		if victim == nil {
			return false
		}
		scheduler.preempt(victim)
	}
}

func (scheduler *Scheduler) hasPreemptedWaiting() bool {
	for _, sequence := range scheduler.waiting {
		if sequence.preempted > 0 {
			return true
		}
	}
	return false
}

func (scheduler *Scheduler) preemptionVictim() *Sequence {
	var victim *Sequence
	for _, sequence := range scheduler.running {
		if sequence.Solo {
			continue
		}
		if victim == nil || sequence.reservedBlocks > victim.reservedBlocks ||
			(sequence.reservedBlocks == victim.reservedBlocks && len(sequence.Tokens) > len(victim.Tokens)) {
			victim = sequence
		}
	}
	return victim
}

func (scheduler *Scheduler) preempt(sequence *Sequence) {
	if sequence.Cache == scheduler.workCache && sequence.Cache != nil {
		scheduler.workCache.ReleaseRow(sequence.Row)
		if sequence.Row >= 0 && sequence.Row < len(scheduler.rows) {
			scheduler.rows[sequence.Row] = nil
		}
	} else if sequence.Cache != nil {
		sequence.Cache.Release()
	}
	sequence.Cache = nil
	sequence.Row = -1
	sequence.Solo = false
	scheduler.reserved -= sequence.reservedBlocks
	if scheduler.reserved < 0 {
		scheduler.reserved = 0
	}
	sequence.reservedBlocks = 0
	sequence.Prefilled = 0
	sequence.lastLogits = nil
	sequence.Status = SequencePrefill
	sequence.preempted++
	scheduler.stats.Preemptions++
	for index, candidate := range scheduler.running {
		if candidate == sequence {
			scheduler.running = append(scheduler.running[:index], scheduler.running[index+1:]...)
			break
		}
	}
	scheduler.waiting = append([]*Sequence{sequence}, scheduler.waiting...)
	scheduler.compactRows()
}

func (scheduler *Scheduler) freeRow() int {
	for index, sequence := range scheduler.rows {
		if sequence == nil {
			return index
		}
	}
	return -1
}

func (scheduler *Scheduler) blocksFor(capacity int) int {
	blocks := (capacity + model.KVBlockSize - 1) / model.KVBlockSize
	if blocks < 1 {
		blocks = 1
	}
	return blocks + 8
}

func (scheduler *Scheduler) speculativeEligible(sequence *Sequence) bool {
	engine := scheduler.Engine
	if !engine.Speculative || (engine.DFlash == nil && engine.DSpark == nil && engine.Drafter == nil) {
		return false
	}
	if sequence.request.Temperature > 0 || sequence.request.Constraint != nil || sequence.request.Options.Constraint != nil {
		return false
	}
	if sequence.request.Options.Logprobs || sequence.request.Options.Thinking || sequence.request.Options.ThinkingBudget > 0 {
		return false
	}
	if len(sequence.request.Stop) > 0 || engine.Model.Config.CEDEnabled() {
		return false
	}
	return sequence.Prefilled == 0
}

// runSpeculative drives the engine's exact greedy speculative loop against the
// sequence's prefill cache. It runs to completion in one scheduler step and is
// used for requests that cannot safely join the batched decode group.
func (scheduler *Scheduler) runSpeculative(sequence *Sequence) []Event {
	var events []Event
	scheduler.Engine.speculativeGenerateWithCache(sequence.Tokens, sequence.request.MaxTokens, sequence.Cache, func(column, mask []int) bool {
		if len(column) == 0 {
			return true
		}
		token := column[0]
		tokenMask := 1
		if len(mask) > 0 {
			tokenMask = mask[0]
		}
		sequence.Generated = append(sequence.Generated, token)
		sequence.Masks = append(sequence.Masks, tokenMask)
		sequence.Tokens = append(sequence.Tokens, token)
		sequence.Position = len(sequence.Tokens)
		events = append(events, Event{Sequence: sequence, Token: token, TokenMask: tokenMask})
		return true
	})
	sequence.Status = SequenceFinished
	if sequence.request.MaxTokens > 0 && len(sequence.Generated) >= sequence.request.MaxTokens {
		sequence.FinishReason = "length"
	} else {
		sequence.FinishReason = "stop"
	}
	events = append(events, Event{Sequence: sequence, Finished: true, FinishReason: sequence.FinishReason})
	return events
}

func (scheduler *Scheduler) prefill(sequence *Sequence) {
	config := scheduler.Engine.Model.Config
	if config.CEDEnabled() {
		ids := tensors.NewInt32sWithData([]int{1, len(sequence.Tokens)}, toI32(sequence.Tokens))
		logits := scheduler.Engine.Model.PrefillCED(ids, sequence.Cache)
		scheduler.stats.PrefillTokens += int64(len(sequence.Tokens))
		sequence.Prefilled = len(sequence.Tokens)
		sequence.lastLogits = lastLogits(logits, config.VocabSize)
	} else {
		for start := sequence.Prefilled; start < len(sequence.Tokens); {
			end := len(sequence.Tokens)
			if scheduler.PrefillChunk > 0 && start+scheduler.PrefillChunk < end {
				end = start + scheduler.PrefillChunk
			}
			if scheduler.Prefix != nil {
				position := sequence.Cache.Position()
				boundary := ((position / model.KVBlockSize) + 1) * model.KVBlockSize
				if boundary > position && boundary-position < end-start {
					end = start + (boundary - position)
				}
			}
			ids := tensors.NewInt32sWithData([]int{1, end - start}, toI32(sequence.Tokens[start:end]))
			logits := scheduler.Engine.Model.Forward(ids, sequence.Cache)
			sequence.Prefilled = end
			sequence.lastLogits = lastLogits(logits, config.VocabSize)
			start = end
			if scheduler.Prefix != nil {
				position := sequence.Cache.Position()
				if position > 0 && position%model.KVBlockSize == 0 {
					scheduler.Prefix.Store(sequence.Tokens[:position], sequence.Cache)
				}
			}
		}
	}
	sequence.Status = SequenceDecode
}

func (scheduler *Scheduler) emit(sequence *Sequence) []Event {
	return emitSequence(scheduler.Engine.Tokenizer, scheduler.Engine.Tools, scheduler.Engine.Model.Config.VocabSize, sequence)
}

func emitSequence(tokenizerImpl *tokenizer.Tokenizer, tools *Registry, vocab int, sequence *Sequence) []Event {
	tokenMask := 1
	var token int
	var logprob *TokenLogprob
	if len(sequence.forcedTokens) > 0 {
		token = sequence.forcedTokens[0]
		sequence.forcedTokens = sequence.forcedTokens[1:]
		tokenMask = 0
	} else {
		var allowed []bool
		if sequence.constraint != nil {
			stopIDs := []int{tokenizerImpl.EncodeSpecial("<|assistant_end|>"), tokenizerImpl.BOSTokenID()}
			allowed = ConstraintMaskCached(tokenizerImpl, sequence.constraint, vocab, stopIDs)
		}
		sampled := SampleNextTokenWithOptions(
			tensors.NewWithData([]int{1, vocab}, sequence.lastLogits),
			sequence.rng,
			SamplingOptions{
				Temperature: sequence.request.Temperature,
				TopK:        sequence.request.TopK,
				TopP:        sequence.request.TopP,
				MinP:        sequence.request.MinP,
				Allowed:     allowed,
			},
		)
		token = int(sampled.Data[0])
		if sequence.request.Options.Logprobs && !isTerminalToken(tokenizerImpl, token) {
			entry := computeTokenLogprob(sequence.lastLogits, token, sequence.request.Temperature, sequence.request.Options.TopLogprobs)
			sequence.Logprobs = append(sequence.Logprobs, entry)
			logprob = &entry
			if sequence.request.Options.OnLogprobs != nil {
				sequence.request.Options.OnLogprobs([]TokenLogprob{entry})
			}
		}
	}
	terminal := token == tokenizerImpl.EncodeSpecial("<|assistant_end|>") || token == tokenizerImpl.BOSTokenID()
	if sequence.constraint != nil && !tokenizerImpl.IsSpecial(token) {
		sequence.constraint.Advance(tokenizerImpl.DecodeSingleTokenBytes(token))
	}
	if !terminal {
		sequence.Generated = append(sequence.Generated, token)
		sequence.Masks = append(sequence.Masks, tokenMask)
	}
	sequence.Tokens = append(sequence.Tokens, token)
	sequence.Position = len(sequence.Tokens)

	toolStart := tokenizerImpl.EncodeSpecial("<|tool_start|>")
	toolEnd := tokenizerImpl.EncodeSpecial("<|tool_end|>")
	switch {
	case token == toolStart:
		sequence.inToolCall = true
		sequence.toolCallTokens = nil
	case token == toolEnd && sequence.inToolCall:
		sequence.inToolCall = false
		if !sequence.request.Options.ExternalTools {
			completeToolCall(tools, tokenizerImpl, sequence)
		}
	case sequence.inToolCall:
		sequence.toolCallTokens = append(sequence.toolCallTokens, token)
	}

	if sequence.request.Options.Thinking || sequence.request.Options.ThinkingBudget > 0 {
		thinkStart := tokenizerImpl.EncodeSpecial("<|think_start|>")
		thinkEnd := tokenizerImpl.EncodeSpecial("<|think_end|>")
		switch {
		case token == thinkStart:
			sequence.inThinking = true
			sequence.thinkingTokens = 0
		case token == thinkEnd && sequence.inThinking:
			sequence.inThinking = false
		case sequence.inThinking:
			sequence.thinkingTokens++
			if sequence.request.Options.ThinkingBudget > 0 && sequence.thinkingTokens >= sequence.request.Options.ThinkingBudget {
				sequence.forcedTokens = append(sequence.forcedTokens, thinkEnd)
			}
		}
	}

	finish := ""
	if terminal {
		finish = "stop"
	} else if sequence.request.MaxTokens > 0 && len(sequence.Generated) >= sequence.request.MaxTokens {
		finish = "length"
	} else {
		text := tokenizerImpl.DecodeSingleTokenBytes(token)
		sequence.content.Write(text)
		if _, stopped := applyStopString(sequence.content.String(), sequence.request.Stop); stopped {
			finish = "stop"
		}
	}
	events := []Event{{Sequence: sequence, Token: token, TokenMask: tokenMask, Logprob: logprob}}
	if finish != "" {
		sequence.Status = SequenceFinished
		sequence.FinishReason = finish
		events = append(events, Event{Sequence: sequence, Finished: true, FinishReason: finish})
	}
	return events
}

func isTerminalToken(tokenizerImpl *tokenizer.Tokenizer, token int) bool {
	return token == tokenizerImpl.EncodeSpecial("<|assistant_end|>") || token == tokenizerImpl.BOSTokenID()
}

func completeToolCall(tools *Registry, tokenizerImpl *tokenizer.Tokenizer, sequence *Sequence) {
	if len(sequence.toolCallTokens) == 0 || tools == nil {
		sequence.toolCallTokens = nil
		return
	}
	expression := tokenizerImpl.Decode(sequence.toolCallTokens)
	sequence.toolCallTokens = nil
	result, ok := tools.Execute(expression)
	if !ok {
		return
	}
	sequence.forcedTokens = append(sequence.forcedTokens, tokenizerImpl.EncodeSpecial("<|tool_output_start|>"))
	sequence.forcedTokens = append(sequence.forcedTokens, tokenizerImpl.Encode(result)...)
	sequence.forcedTokens = append(sequence.forcedTokens, tokenizerImpl.EncodeSpecial("<|tool_output_end|>"))
}

func (scheduler *Scheduler) reap() {
	active := scheduler.running[:0]
	for _, sequence := range scheduler.running {
		if sequence.Status == SequenceFinished || sequence.Status == SequenceFailed {
			if sequence.Cache == scheduler.workCache && sequence.Cache != nil {
				scheduler.workCache.ReleaseRow(sequence.Row)
				if sequence.Row >= 0 && sequence.Row < len(scheduler.rows) {
					scheduler.rows[sequence.Row] = nil
				}
			} else if sequence.Cache != nil {
				sequence.Cache.Release()
			}
			sequence.Cache = nil
			sequence.Row = -1
			sequence.Solo = false
			scheduler.reserved -= sequence.reservedBlocks
			if scheduler.reserved < 0 {
				scheduler.reserved = 0
			}
			sequence.reservedBlocks = 0
			scheduler.finished = append(scheduler.finished, sequence)
			continue
		}
		active = append(active, sequence)
	}
	scheduler.running = active
}

func (scheduler *Scheduler) compactRows() {
	if scheduler.workCache == nil {
		return
	}
	for free := 0; free < len(scheduler.rows); free++ {
		if scheduler.rows[free] != nil {
			continue
		}
		high := -1
		for index := len(scheduler.rows) - 1; index > free; index-- {
			if scheduler.rows[index] != nil {
				high = index
				break
			}
		}
		if high < 0 {
			break
		}
		sequence := scheduler.rows[high]
		if !scheduler.workCache.MoveRow(free, high) {
			break
		}
		scheduler.rows[free] = sequence
		scheduler.rows[high] = nil
		sequence.Row = free
	}
}

func (scheduler *Scheduler) activeRows() []*Sequence {
	active := make([]*Sequence, 0, len(scheduler.rows))
	for _, sequence := range scheduler.rows {
		if sequence != nil {
			active = append(active, sequence)
		}
	}
	return active
}

func cloneConstraint(constraint Constraint) Constraint {
	if constraint == nil {
		return nil
	}
	return constraint.Clone()
}

func lastLogits(logits *tensors.Tensor, vocab int) []float32 {
	rows := logits.Numel() / vocab
	return logitsRow(logits, rows-1, vocab)
}

func logitsRow(logits *tensors.Tensor, row, vocab int) []float32 {
	base := row * vocab
	return append([]float32(nil), logits.Data[base:base+vocab]...)
}

func applyStopString(content string, stops []string) (string, bool) {
	cutoff := -1
	for _, stop := range stops {
		if stop == "" {
			continue
		}
		if index := strings.Index(content, stop); index >= 0 && (cutoff < 0 || index < cutoff) {
			cutoff = index
		}
	}
	if cutoff < 0 {
		return content, false
	}
	return content[:cutoff], true
}

// SortFinished sorts finished sequences by ID for deterministic tests.
func SortFinished(sequences []*Sequence) {
	sort.SliceStable(sequences, func(left, right int) bool { return sequences[left].ID < sequences[right].ID })
}
