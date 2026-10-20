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
	// Row is the sequence's row in the prefill or decode cache; -1 when the
	// sequence is not admitted or is running solo.
	Row int
	// Solo marks a request that cannot join the batched paths (speculative
	// decoding) and owns a private cache.
	Solo bool
	// Logprobs holds one entry per sampled token when requested.
	Logprobs []TokenLogprob

	reservedBlocks int
	memberReserved []int
	preempted      int
	weights        []float32
	forcedTokens   []int
	inToolCall     bool
	toolCallTokens []int
	inThinking     bool
	thinkingTokens int

	request        Request
	rng            *tensors.RNG
	lastLogits     []float32
	specLastHidden *tensors.Tensor
	specPending    int
	specHasPending bool
	content        strings.Builder
	constraint     Constraint
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
// sequences. Prefill rows live in a separate cache so they can advance in
// chunks while decode rows are active; completed rows are adopted by the decode
// cache without copying their KV. Both caches share one paged block manager.
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

	waiting      []*Sequence
	running      []*Sequence
	finished     []*Sequence
	workCache    *model.KVBuffer
	prefillCache *model.KVBuffer
	cedCache     *model.KVBuffer
	specCache    *model.KVBuffer
	rows         []*Sequence
	prefillRows  []*Sequence
	pools        *model.KVBlockPools
	reserved     int
	nextSeed     uint64
	stats        SchedulerStats

	// Batch counters are used by parity tests to confirm grouped execution.
	cedBatches          int
	speculativeForwards int
	speculativeRows     int
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
		scheduler.releaseSequence(sequence)
		sequence.Status = SequenceFailed
		scheduler.running = append(scheduler.running[:index], scheduler.running[index+1:]...)
		scheduler.compactRows()
		scheduler.compactPrefillRows()
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

// Step admits waiting requests, advances detached prefill rows by one chunk and
// decode rows by one token. Splitting the caches means an active decode batch no
// longer forces prefills to advance one token per step.
func (scheduler *Scheduler) Step() []Event {
	scheduler.ensureCaches()
	scheduler.admit()

	events := []Event{}
	sampled := make(map[*Sequence]bool)

	for _, sequence := range scheduler.running {
		if sequence.Status != SequencePrefill || !sequence.Solo {
			continue
		}
		if scheduler.speculativeEligible(sequence) {
			scheduler.prefillSpeculative(sequence)
		}
	}

	prefillEvents, promoted := scheduler.prefillStep()
	events = append(events, prefillEvents...)
	for sequence := range promoted {
		sampled[sequence] = true
	}

	for _, sequence := range scheduler.running {
		if sequence.Status != SequenceDecode || sequence.Solo || sampled[sequence] {
			continue
		}
		events = append(events, scheduler.emit(sequence)...)
	}

	scheduler.reap()
	scheduler.compactRows()
	scheduler.compactPrefillRows()

	scheduler.decodeForward()
	events = append(events, scheduler.speculativeForward()...)
	scheduler.reap()
	scheduler.compactRows()
	scheduler.compactPrefillRows()

	for _, event := range events {
		if event.TokenMask == 1 {
			scheduler.stats.Generated++
		}
	}
	return events
}

// prefillStep advances every detached prefill row. CED rows are grouped by
// prompt length and prefilled together; normal rows advance by one common
// chunk, even while decode rows are active.
func (scheduler *Scheduler) prefillStep() ([]Event, map[*Sequence]bool) {
	events := []Event{}
	promoted := make(map[*Sequence]bool)
	scheduler.compactPrefillRows()
	config := scheduler.Engine.Model.Config

	if config.CEDEnabled() {
		scheduler.cedPrefillStep(&events, promoted)
		return events, promoted
	}

	// Retry promotions left over from an earlier step.
	for _, sequence := range scheduler.prefillRows {
		if sequence == nil || sequence.Status != SequencePrefill || sequence.Prefilled < len(sequence.Tokens) {
			continue
		}
		if scheduler.promote(sequence) {
			events = append(events, scheduler.emit(sequence)...)
			promoted[sequence] = true
		}
	}
	// Promotions above can leave holes; batch forwards index cache rows by
	// position, so compact before building the active row set.
	scheduler.compactPrefillRows()

	active := make([]*Sequence, 0, len(scheduler.prefillRows))
	for _, sequence := range scheduler.prefillRows {
		if sequence == nil || sequence.Status != SequencePrefill || sequence.Prefilled >= len(sequence.Tokens) {
			continue
		}
		active = append(active, sequence)
	}
	if len(active) == 0 {
		return events, promoted
	}
	chunk := 0
	for _, sequence := range active {
		remaining := len(sequence.Tokens) - sequence.Prefilled
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
			position := scheduler.prefillCache.PositionAt(sequence.Row)
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
	logits := scheduler.Engine.Model.Forward(ids, scheduler.prefillCache)
	vocab := config.VocabSize
	for index, sequence := range active {
		sequence.Prefilled += chunk
		sequence.lastLogits = logitsRow(logits, index*chunk+chunk-1, vocab)
		position := scheduler.prefillCache.PositionAt(sequence.Row)
		if scheduler.Prefix != nil && position > 0 && position%model.KVBlockSize == 0 {
			if snapshot := scheduler.prefillCache.CloneRow(sequence.Row); snapshot != nil {
				scheduler.Prefix.Store(sequence.Tokens[:position], snapshot)
				snapshot.Release()
			}
		}
		if sequence.Prefilled == len(sequence.Tokens) {
			if scheduler.promote(sequence) {
				events = append(events, scheduler.emit(sequence)...)
				promoted[sequence] = true
			}
		}
	}
	return events, promoted
}

// cedPrefillStep batches CED prefills that share a token length. CED's encoder
// must see the full prompt, and padding would contaminate compressed blocks, so
// different lengths run as separate batches.
func (scheduler *Scheduler) cedPrefillStep(events *[]Event, promoted map[*Sequence]bool) {
	type prefillGroup struct {
		length    int
		sequences []*Sequence
	}
	groups := make([]prefillGroup, 0)
	groupIndex := make(map[int]int)
	for _, sequence := range scheduler.prefillRows {
		if sequence == nil || sequence.Status != SequencePrefill || sequence.Prefilled >= len(sequence.Tokens) {
			continue
		}
		length := len(sequence.Tokens)
		index, ok := groupIndex[length]
		if !ok {
			index = len(groups)
			groupIndex[length] = index
			groups = append(groups, prefillGroup{length: length})
		}
		groups[index].sequences = append(groups[index].sequences, sequence)
	}
	config := scheduler.Engine.Model.Config
	for _, group := range groups {
		count := len(group.sequences)
		for row := 0; row < count; row++ {
			scheduler.cedCache.ReleaseRow(row)
		}
		data := make([]int32, count*group.length)
		for row, sequence := range group.sequences {
			copy(data[row*group.length:(row+1)*group.length], toI32(sequence.Tokens))
		}
		ids := tensors.NewInt32sWithData([]int{count, group.length}, data)
		logits := scheduler.Engine.Model.PrefillCED(ids, scheduler.cedCache)
		scheduler.cedBatches++
		scheduler.stats.PrefillTokens += int64(count * group.length)
		failed := false
		for row, sequence := range group.sequences {
			if failed {
				sequence.Prefilled = 0
				sequence.lastLogits = nil
				continue
			}
			sequence.Prefilled = group.length
			sequence.lastLogits = logitsRow(logits, row*group.length+group.length-1, config.VocabSize)
			if !scheduler.promoteCEDAt(sequence, row) {
				sequence.Prefilled = 0
				sequence.lastLogits = nil
				failed = true
				continue
			}
			*events = append(*events, scheduler.emit(sequence)...)
			promoted[sequence] = true
		}
		for row := 0; row < count; row++ {
			scheduler.cedCache.ReleaseRow(row)
		}
	}
}

// promote moves a completed prefill row into the decode cache without copying
// KV blocks, then frees the prefill row.
func (scheduler *Scheduler) promote(sequence *Sequence) bool {
	if sequence == nil || sequence.Cache != scheduler.prefillCache {
		return false
	}
	row := scheduler.freeRow()
	if row < 0 {
		return false
	}
	prefillRow := sequence.Row
	if !scheduler.workCache.AdoptRowFrom(row, scheduler.prefillCache, prefillRow) {
		return false
	}
	scheduler.prefillCache.ReleaseRow(prefillRow)
	if prefillRow >= 0 && prefillRow < len(scheduler.prefillRows) {
		scheduler.prefillRows[prefillRow] = nil
	}
	sequence.Cache = scheduler.workCache
	sequence.Row = row
	scheduler.rows[row] = sequence
	sequence.Status = SequenceDecode
	return true
}

// promoteCEDAt moves one CED batch row into the decode cache and releases its
// prefill marker row.
func (scheduler *Scheduler) promoteCEDAt(sequence *Sequence, sourceRow int) bool {
	if sequence == nil || scheduler.cedCache == nil || sequence.Cache != scheduler.prefillCache {
		return false
	}
	row := scheduler.freeRow()
	if row < 0 {
		return false
	}
	if !scheduler.workCache.AdoptRowFrom(row, scheduler.cedCache, sourceRow) {
		return false
	}
	scheduler.cedCache.ReleaseRow(sourceRow)
	prefillRow := sequence.Row
	scheduler.prefillCache.ReleaseRow(prefillRow)
	if prefillRow >= 0 && prefillRow < len(scheduler.prefillRows) {
		scheduler.prefillRows[prefillRow] = nil
	}
	sequence.Cache = scheduler.workCache
	sequence.Row = row
	scheduler.rows[row] = sequence
	sequence.Status = SequenceDecode
	return true
}

// decodeForward advances every decode row by its sampled token in one forward.
func (scheduler *Scheduler) decodeForward() {
	active := scheduler.activeRows()
	if len(active) == 0 {
		return
	}
	tokens := make([]int32, 0, len(active))
	positions := make([]*Sequence, 0, len(active))
	for _, sequence := range active {
		if sequence.Status != SequenceDecode {
			continue
		}
		tokens = append(tokens, int32(sequence.Tokens[len(sequence.Tokens)-1]))
		positions = append(positions, sequence)
		scheduler.stats.DecodeTokens++
	}
	if len(positions) == 0 {
		return
	}
	ids := tensors.NewInt32sWithData([]int{len(positions), 1}, tokens)
	logits := scheduler.Engine.Model.Forward(ids, scheduler.workCache)
	vocab := scheduler.Engine.Model.Config.VocabSize
	for index, sequence := range positions {
		sequence.lastLogits = logitsRow(logits, index, vocab)
	}
}

func (scheduler *Scheduler) ensureCaches() {
	if scheduler.workCache != nil && scheduler.prefillCache != nil {
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
		// Both caches share this manager, but a sequence only ever occupies a
		// row in one of them, so the budget remains one row per active request.
		scheduler.Blocks = model.NewKVBlockManager(maxRows*(blocksPerRow+8) + extra)
	}
	scheduler.pools = model.NewKVBlockPools(scheduler.Blocks, config.NumLayer, config.NumKVHead, config.HeadDim())
	scheduler.workCache = newCacheForModelWithPools(scheduler.Blocks, scheduler.pools, config, maxRows, config.SequenceLen)
	scheduler.prefillCache = newCacheForModelWithPools(scheduler.Blocks, scheduler.pools, config, maxRows, config.SequenceLen)
	if config.CEDEnabled() {
		// CED prefills are grouped by prompt length and run through this
		// scratch cache before joining the decode cache.
		scheduler.cedCache = newCacheForModelWithPools(scheduler.Blocks, scheduler.pools, config, maxRows, config.SequenceLen)
	}
	if scheduler.Engine.Speculative &&
		(scheduler.Engine.DFlash != nil || scheduler.Engine.DSpark != nil || scheduler.Engine.Drafter != nil) {
		scheduler.specCache = newCacheForModelWithPools(scheduler.Blocks, scheduler.pools, config, maxRows, config.SequenceLen)
	}
	scheduler.rows = make([]*Sequence, maxRows)
	scheduler.prefillRows = make([]*Sequence, maxRows)
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

		if scheduler.speculativeEligible(sequence) {
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

		row := scheduler.freePrefillRow()
		if row < 0 {
			scheduler.waiting = append([]*Sequence{sequence}, scheduler.waiting...)
			break
		}
		needed := scheduler.blocksFor(capacity)
		var snapshot *model.KVBuffer
		matched := 0
		if scheduler.Prefix != nil && !config.CEDEnabled() {
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
		sequence.Cache = scheduler.prefillCache
		sequence.Row = row
		scheduler.prefillRows[row] = sequence
		scheduler.prefillCache.SetPositionAt(row, 0)
		if snapshot != nil {
			if scheduler.prefillCache.AdoptRow(row, snapshot) {
				sequence.Prefilled = matched
			} else {
				additional := scheduler.blocksFor(capacity) - needed
				scheduler.reserved += additional
				sequence.reservedBlocks += additional
			}
			snapshot.Release()
		}
		sequence.Solo = false
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

// releaseSequence frees whichever cache row the sequence owns and returns its
// reserved block budget.
func (scheduler *Scheduler) releaseSequence(sequence *Sequence) {
	switch {
	case sequence.Cache == nil:
	case sequence.Cache == scheduler.workCache:
		scheduler.workCache.ReleaseRow(sequence.Row)
		if sequence.Row >= 0 && sequence.Row < len(scheduler.rows) {
			scheduler.rows[sequence.Row] = nil
		}
	case sequence.Cache == scheduler.prefillCache:
		scheduler.prefillCache.ReleaseRow(sequence.Row)
		if sequence.Row >= 0 && sequence.Row < len(scheduler.prefillRows) {
			scheduler.prefillRows[sequence.Row] = nil
		}
	default:
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
}

func (scheduler *Scheduler) preempt(sequence *Sequence) {
	scheduler.releaseSequence(sequence)
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
	scheduler.compactPrefillRows()
}

func (scheduler *Scheduler) freeRow() int {
	for index, sequence := range scheduler.rows {
		if sequence == nil {
			return index
		}
	}
	return -1
}

func (scheduler *Scheduler) freePrefillRow() int {
	for index, sequence := range scheduler.prefillRows {
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

// prefillSpeculative loads the prompt into a speculative request's private
// cache and captures the target hidden state DFlash needs for drafting.
func (scheduler *Scheduler) prefillSpeculative(sequence *Sequence) {
	config := scheduler.Engine.Model.Config
	ids := tensors.NewInt32sWithData([]int{1, len(sequence.Tokens)}, toI32(sequence.Tokens))
	var logits *tensors.Tensor
	if scheduler.Engine.DFlash != nil {
		var hidden *tensors.Tensor
		logits, hidden = scheduler.Engine.Model.ForwardHiddenAtLayers(ids, sequence.Cache, scheduler.Engine.DFlash.TargetIDs)
		sequence.specLastHidden = lastHiddenRow(hidden)
	} else {
		logits = scheduler.Engine.Model.Forward(ids, sequence.Cache)
	}
	scheduler.stats.PrefillTokens += int64(len(sequence.Tokens))
	sequence.Prefilled = len(sequence.Tokens)
	sequence.lastLogits = lastLogits(logits, config.VocabSize)
	sequence.Status = SequenceDecode
}

func finishSpeculative(sequence *Sequence, reason string) []Event {
	sequence.Status = SequenceFinished
	sequence.FinishReason = reason
	return []Event{{Sequence: sequence, Finished: true, FinishReason: reason}}
}

func (scheduler *Scheduler) terminalToken(token int) bool {
	return token == scheduler.Engine.Tokenizer.EncodeSpecial("<|assistant_end|>") || token == scheduler.Engine.Tokenizer.BOSTokenID()
}

// speculativeForward verifies one draft block for every speculative decode row
// in a single target forward. Rows with unequal draft lengths are padded; rows
// that needed padding or partial acceptance are rolled back and replayed over
// only their accepted prefix.
func (scheduler *Scheduler) speculativeForward() []Event {
	if scheduler.specCache == nil {
		return nil
	}
	rows := make([]*Sequence, 0, len(scheduler.running))
	for _, sequence := range scheduler.running {
		if sequence.Solo && sequence.Status == SequenceDecode {
			rows = append(rows, sequence)
		}
	}
	if len(rows) == 0 {
		return nil
	}
	config := scheduler.Engine.Model.Config
	draftLength := scheduler.Engine.DraftLength
	if draftLength < 2 {
		draftLength = defaultDraftLength
	}

	type speculativeRow struct {
		sequence   *Sequence
		candidates []int
		mark       model.KVMark
		base       int
	}
	prepared := make([]speculativeRow, 0, len(rows))
	events := []Event{}
	maxLength := 0
	for _, sequence := range rows {
		base := sequence.Cache.Position()
		room := config.SequenceLen - base
		remaining := sequence.request.MaxTokens - len(sequence.Generated)
		if room <= 0 || remaining <= 0 {
			reason := "stop"
			if remaining <= 0 {
				reason = "length"
			}
			events = append(events, finishSpeculative(sequence, reason)...)
			continue
		}
		first := sequence.specPending
		if !sequence.specHasPending {
			first = argmaxSlice(sequence.lastLogits)
		}
		sequence.specHasPending = false
		blockSize := draftLength
		if blockSize > room {
			blockSize = room
		}
		if blockSize > remaining {
			blockSize = remaining
		}
		if blockSize < 1 {
			blockSize = 1
		}
		candidates := scheduler.Engine.draftCandidates(sequence.Tokens, first, sequence.specLastHidden, blockSize-1)
		if len(candidates) > remaining {
			candidates = candidates[:remaining]
		}
		if len(candidates) > room {
			candidates = candidates[:room]
		}
		if len(candidates) == 0 {
			events = append(events, finishSpeculative(sequence, "length")...)
			continue
		}
		row := len(prepared)
		mark := sequence.Cache.Mark()
		if !scheduler.specCache.AdoptRowFrom(row, sequence.Cache, 0) {
			continue
		}
		prepared = append(prepared, speculativeRow{sequence: sequence, candidates: candidates, mark: mark, base: base})
		if len(candidates) > maxLength {
			maxLength = len(candidates)
		}
	}
	if len(prepared) == 0 {
		return events
	}
	scheduler.speculativeForwards++
	scheduler.speculativeRows += len(prepared)

	data := make([]int32, len(prepared)*maxLength)
	for row, current := range prepared {
		copy(data[row*maxLength:(row+1)*maxLength], toI32(current.candidates))
	}
	ids := tensors.NewInt32sWithData([]int{len(prepared), maxLength}, data)
	var verified *tensors.Tensor
	var verifiedHidden *tensors.Tensor
	if scheduler.Engine.DFlash != nil {
		verified, verifiedHidden = scheduler.Engine.Model.ForwardHiddenAtLayers(ids, scheduler.specCache, scheduler.Engine.DFlash.TargetIDs)
	} else {
		verified = scheduler.Engine.Model.Forward(ids, scheduler.specCache)
	}

	for row, current := range prepared {
		sequence := current.sequence
		blockSize := len(current.candidates)
		matched := 0
		for matched < blockSize-1 && argmaxRow(verified, row*maxLength+matched) == current.candidates[matched+1] {
			matched++
		}
		accepted := 1 + matched
		if !sequence.Cache.AdoptRowFrom(0, scheduler.specCache, row) {
			scheduler.specCache.ReleaseRow(row)
			continue
		}
		needsReplay := accepted < blockSize || blockSize < maxLength
		if needsReplay {
			sequence.Cache.Rollback(current.mark)
			acceptedIDs := tensors.NewInt32sWithData([]int{1, accepted}, toI32(current.candidates[:accepted]))
			var replayHidden *tensors.Tensor
			if scheduler.Engine.DFlash != nil {
				_, replayHidden = scheduler.Engine.Model.ForwardHiddenAtLayers(acceptedIDs, sequence.Cache, scheduler.Engine.DFlash.TargetIDs)
				sequence.specLastHidden = lastHiddenRow(replayHidden)
			} else {
				scheduler.Engine.Model.Forward(acceptedIDs, sequence.Cache)
			}
		} else if scheduler.Engine.DFlash != nil && verifiedHidden != nil {
			sequence.specLastHidden = hiddenRow(verifiedHidden, row*maxLength+accepted-1)
		}
		sequence.Cache.SetPosition(current.base + accepted)
		scheduler.Engine.Model.SetPreviousEmbeddingForToken(sequence.Cache, current.candidates[accepted-1])
		scheduler.specCache.ReleaseRow(row)
		scheduler.stats.DecodeTokens += int64(accepted)

		var next int
		if matched < blockSize-1 {
			next = argmaxRow(verified, row*maxLength+matched)
		} else {
			next = argmaxRow(verified, row*maxLength+blockSize-1)
		}
		sequence.specPending = next
		sequence.specHasPending = true

		finished := false
		for index := 0; index < accepted; index++ {
			token := current.candidates[index]
			if scheduler.terminalToken(token) {
				events = append(events, finishSpeculative(sequence, "stop")...)
				finished = true
				break
			}
			sequence.Generated = append(sequence.Generated, token)
			sequence.Masks = append(sequence.Masks, 1)
			sequence.Tokens = append(sequence.Tokens, token)
			sequence.Position = len(sequence.Tokens)
			events = append(events, Event{Sequence: sequence, Token: token, TokenMask: 1})
			if sequence.request.MaxTokens > 0 && len(sequence.Generated) >= sequence.request.MaxTokens {
				events = append(events, finishSpeculative(sequence, "length")...)
				finished = true
				break
			}
		}
		if finished {
			sequence.specHasPending = false
		}
	}
	return events
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
			scheduler.releaseSequence(sequence)
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

func (scheduler *Scheduler) compactPrefillRows() {
	if scheduler.prefillCache == nil {
		return
	}
	for free := 0; free < len(scheduler.prefillRows); free++ {
		if scheduler.prefillRows[free] != nil {
			continue
		}
		high := -1
		for index := len(scheduler.prefillRows) - 1; index > free; index-- {
			if scheduler.prefillRows[index] != nil {
				high = index
				break
			}
		}
		if high < 0 {
			break
		}
		sequence := scheduler.prefillRows[high]
		if !scheduler.prefillCache.MoveRow(free, high) {
			break
		}
		scheduler.prefillRows[free] = sequence
		scheduler.prefillRows[high] = nil
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
