package inference

import (
	"sort"

	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tensors"
)

// EnsembleScheduler batches independent requests that share one Ensemble. It
// keeps detached prefill and decode caches per member, advances the same rows
// through every member, and samples once from the weighted-sum logits.
type EnsembleScheduler struct {
	Ensemble     *Ensemble
	MaxActive    int
	PrefillChunk int
	// Preemption selects whether admission may evict a running request.
	Preemption PreemptionPolicy
	// Blocks optionally supplies one shared paged block pool per member. When
	// empty, each member gets a pool sized from MaxActive and its sequence
	// length. Prefill and decode caches for a member share its pool.
	Blocks []*model.KVBlockManager

	waiting  []*Sequence
	running  []*Sequence
	finished []*Sequence

	rows          []*Sequence
	prefillRows   []*Sequence
	caches        []*model.KVBuffer
	prefillCaches []*model.KVBuffer
	cedCaches     []*model.KVBuffer
	pools         []*model.KVBlockPools
	weights       []float32
	vocab         int
	reserved      []int
	nextSeed      uint64
	stats         SchedulerStats
}

// NewEnsembleScheduler builds a batched scheduler over an ensemble.
func NewEnsembleScheduler(ensemble *Ensemble, maxActive int) *EnsembleScheduler {
	return &EnsembleScheduler{Ensemble: ensemble, MaxActive: maxActive, nextSeed: 1}
}

// Submit queues a request.
func (scheduler *EnsembleScheduler) Submit(request Request) *Sequence {
	if scheduler.Ensemble == nil || len(scheduler.Ensemble.Models) == 0 {
		panic("inference: ensemble scheduler has no models")
	}
	configuration := scheduler.Ensemble.Models[0].Model.Config
	if request.MaxTokens <= 0 {
		request.MaxTokens = configuration.SequenceLen
	}
	seed := request.Seed
	if seed == 0 {
		seed = scheduler.nextSeed
		scheduler.nextSeed++
	}
	weights := scheduler.Ensemble.normalizedWeights()
	if len(request.Weights) == len(weights) {
		weights = normalizeWeightValues(request.Weights)
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
		weights:    weights,
		constraint: cloneConstraint(constraint),
	}
	if (request.Options.Thinking || request.Options.ThinkingBudget > 0) && len(request.Prompt) > 0 {
		thinkStart := scheduler.Ensemble.Tokenizer.EncodeSpecial("<|think_start|>")
		if request.Prompt[len(request.Prompt)-1] == thinkStart {
			sequence.inThinking = true
		}
	}
	scheduler.waiting = append(scheduler.waiting, sequence)
	return sequence
}

// Pending reports whether the scheduler still has work.
func (scheduler *EnsembleScheduler) Pending() bool {
	return len(scheduler.waiting) > 0 || len(scheduler.running) > 0
}

// Stats returns a progress snapshot.
func (scheduler *EnsembleScheduler) Stats() SchedulerStats {
	stats := scheduler.stats
	stats.Waiting = len(scheduler.waiting)
	stats.Running = len(scheduler.running)
	stats.Finished = len(scheduler.finished)
	for _, manager := range scheduler.Blocks {
		if manager != nil {
			stats.KVBlocksInUse += manager.InUse()
			stats.KVBlocksFree += manager.FreeBlocks()
		}
	}
	return stats
}

// Run executes until every submitted request finishes.
func (scheduler *EnsembleScheduler) Run() []*Sequence {
	for scheduler.Pending() {
		scheduler.Step()
	}
	return scheduler.finished
}

func (scheduler *EnsembleScheduler) ensureCaches() {
	if len(scheduler.caches) > 0 && len(scheduler.prefillCaches) > 0 {
		return
	}
	maxRows := scheduler.MaxActive
	if maxRows <= 0 {
		maxRows = 8
		scheduler.MaxActive = maxRows
	}
	scheduler.vocab = scheduler.Ensemble.Models[0].Model.Config.VocabSize
	for _, member := range scheduler.Ensemble.Models {
		if member.Model.Config.VocabSize != scheduler.vocab {
			panic("inference: ensemble models must share a vocab size")
		}
	}
	scheduler.weights = scheduler.Ensemble.normalizedWeights()
	memberCount := len(scheduler.Ensemble.Models)
	if len(scheduler.Blocks) != memberCount {
		scheduler.Blocks = make([]*model.KVBlockManager, memberCount)
	}
	scheduler.caches = make([]*model.KVBuffer, memberCount)
	scheduler.prefillCaches = make([]*model.KVBuffer, memberCount)
	scheduler.pools = make([]*model.KVBlockPools, memberCount)
	scheduler.reserved = make([]int, memberCount)
	cedMembers := 0
	for _, member := range scheduler.Ensemble.Models {
		if member.Model.Config.CEDEnabled() {
			cedMembers++
		}
	}
	if cedMembers > 0 && cedMembers != memberCount {
		panic("inference: ensemble cannot mix CED and non-CED models")
	}
	if cedMembers == memberCount {
		scheduler.cedCaches = make([]*model.KVBuffer, memberCount)
	}
	for index, member := range scheduler.Ensemble.Models {
		config := member.Model.Config
		blocksPerRow := (config.SequenceLen + model.KVBlockSize - 1) / model.KVBlockSize
		if blocksPerRow < 1 {
			blocksPerRow = 1
		}
		if scheduler.Blocks[index] == nil {
			scheduler.Blocks[index] = model.NewKVBlockManager(maxRows * (blocksPerRow + 8))
		}
		scheduler.pools[index] = model.NewKVBlockPools(scheduler.Blocks[index], config.NumLayer, config.NumKVHead, config.HeadDim())
		scheduler.caches[index] = newCacheForModelWithPools(scheduler.Blocks[index], scheduler.pools[index], config, maxRows, config.SequenceLen)
		scheduler.prefillCaches[index] = newCacheForModelWithPools(scheduler.Blocks[index], scheduler.pools[index], config, maxRows, config.SequenceLen)
		if scheduler.cedCaches != nil {
			scheduler.cedCaches[index] = newCacheForModelWithPools(scheduler.Blocks[index], scheduler.pools[index], config, maxRows, config.SequenceLen)
		}
	}
	scheduler.rows = make([]*Sequence, maxRows)
	scheduler.prefillRows = make([]*Sequence, maxRows)
}

func (scheduler *EnsembleScheduler) freeRow() int {
	for index, sequence := range scheduler.rows {
		if sequence == nil {
			return index
		}
	}
	return -1
}

func (scheduler *EnsembleScheduler) freePrefillRow() int {
	for index, sequence := range scheduler.prefillRows {
		if sequence == nil {
			return index
		}
	}
	return -1
}

func (scheduler *EnsembleScheduler) activeRows() []*Sequence {
	active := make([]*Sequence, 0, len(scheduler.rows))
	for _, sequence := range scheduler.rows {
		if sequence != nil {
			active = append(active, sequence)
		}
	}
	return active
}

func (scheduler *EnsembleScheduler) admissionCapacity(sequence *Sequence, member int) int {
	config := scheduler.Ensemble.Models[member].Model.Config
	capacity := len(sequence.Prompt) + sequence.request.MaxTokens
	if capacity > config.SequenceLen {
		capacity = config.SequenceLen
	}
	if capacity < len(sequence.Prompt) {
		capacity = len(sequence.Prompt)
	}
	return capacity
}

func (scheduler *EnsembleScheduler) blocksFor(member, capacity int) int {
	blocks := (capacity + model.KVBlockSize - 1) / model.KVBlockSize
	if blocks < 1 {
		blocks = 1
	}
	return blocks + 8
}

func (scheduler *EnsembleScheduler) admit() {
	limit := scheduler.MaxActive
	for len(scheduler.waiting) > 0 && (limit <= 0 || len(scheduler.running) < limit) {
		sequence := scheduler.waiting[0]
		scheduler.waiting = scheduler.waiting[1:]
		row := scheduler.freePrefillRow()
		if row < 0 {
			scheduler.waiting = append([]*Sequence{sequence}, scheduler.waiting...)
			break
		}
		needed := make([]int, len(scheduler.Ensemble.Models))
		for member := range scheduler.Ensemble.Models {
			needed[member] = scheduler.blocksFor(member, scheduler.admissionCapacity(sequence, member))
		}
		if !scheduler.ensureBudget(sequence, needed) {
			scheduler.waiting = append([]*Sequence{sequence}, scheduler.waiting...)
			break
		}
		for member := range needed {
			scheduler.reserved[member] += needed[member]
		}
		sequence.memberReserved = needed
		sequence.Row = row
		sequence.Cache = scheduler.prefillCaches[0]
		for _, cache := range scheduler.prefillCaches {
			cache.SetPositionAt(row, 0)
		}
		sequence.Status = SequencePrefill
		scheduler.prefillRows[row] = sequence
		scheduler.running = append(scheduler.running, sequence)
	}
}

func (scheduler *EnsembleScheduler) ensureBudget(sequence *Sequence, needed []int) bool {
	for {
		fits := true
		for member := range needed {
			if scheduler.Blocks[member] == nil ||
				scheduler.reserved[member]+needed[member] > scheduler.Blocks[member].BlockCount() {
				fits = false
				break
			}
		}
		if fits || len(scheduler.running) == 0 {
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

func (scheduler *EnsembleScheduler) hasPreemptedWaiting() bool {
	for _, sequence := range scheduler.waiting {
		if sequence.preempted > 0 {
			return true
		}
	}
	return false
}

func (scheduler *EnsembleScheduler) preemptionVictim() *Sequence {
	var victim *Sequence
	victimTotal := 0
	for _, sequence := range scheduler.running {
		total := 0
		for _, blocks := range sequence.memberReserved {
			total += blocks
		}
		if victim == nil || total > victimTotal ||
			(total == victimTotal && len(sequence.Tokens) > len(victim.Tokens)) {
			victim = sequence
			victimTotal = total
		}
	}
	return victim
}

// Step admits requests, advances detached prefill rows by one chunk and decode
// rows by one token. Prefills keep chunking while decode rows are active.
func (scheduler *EnsembleScheduler) Step() []Event {
	scheduler.ensureCaches()
	scheduler.admit()
	events := []Event{}
	sampled := make(map[*Sequence]bool)

	prefillEvents, promoted := scheduler.prefillStep()
	events = append(events, prefillEvents...)
	for sequence := range promoted {
		sampled[sequence] = true
	}
	for _, sequence := range scheduler.running {
		if sequence.Status != SequenceDecode || sampled[sequence] {
			continue
		}
		events = append(events, emitSequence(scheduler.Ensemble.Tokenizer, scheduler.Ensemble.Tools, scheduler.vocab, sequence)...)
	}
	scheduler.reap()
	scheduler.compactRows()
	scheduler.compactPrefillRows()
	scheduler.decodeForward()
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

func (scheduler *EnsembleScheduler) prefillStep() ([]Event, map[*Sequence]bool) {
	events := []Event{}
	promoted := make(map[*Sequence]bool)
	scheduler.compactPrefillRows()
	if scheduler.cedCaches != nil {
		scheduler.cedPrefillStep(&events, promoted)
		return events, promoted
	}

	for _, sequence := range scheduler.prefillRows {
		if sequence == nil || sequence.Status != SequencePrefill || sequence.Prefilled < len(sequence.Tokens) {
			continue
		}
		if scheduler.promote(sequence) {
			events = append(events, emitSequence(scheduler.Ensemble.Tokenizer, scheduler.Ensemble.Tools, scheduler.vocab, sequence)...)
			promoted[sequence] = true
		}
	}
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
	data := make([]int32, len(active)*chunk)
	for index, sequence := range active {
		copy(data[index*chunk:(index+1)*chunk], toI32(sequence.Tokens[sequence.Prefilled:sequence.Prefilled+chunk]))
	}
	scheduler.stats.PrefillTokens += int64(len(active) * chunk)
	ids := tensors.NewInt32sWithData([]int{len(active), chunk}, data)
	combined := scheduler.forwardMembers(ids, len(active), chunk, active, scheduler.prefillCaches)
	for index, sequence := range active {
		sequence.Prefilled += chunk
		sequence.lastLogits = logitsRow(combined, index*chunk+chunk-1, scheduler.vocab)
		if sequence.Prefilled == len(sequence.Tokens) {
			if scheduler.promote(sequence) {
				events = append(events, emitSequence(scheduler.Ensemble.Tokenizer, scheduler.Ensemble.Tools, scheduler.vocab, sequence)...)
				promoted[sequence] = true
			}
		}
	}
	return events, promoted
}

// cedPrefillStep batches CED rows that share a prompt length across every
// member. Different lengths run as separate batches because padding would
// contaminate compressed blocks.
func (scheduler *EnsembleScheduler) cedPrefillStep(events *[]Event, promoted map[*Sequence]bool) {
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
	for _, group := range groups {
		count := len(group.sequences)
		for _, cache := range scheduler.cedCaches {
			for row := 0; row < count; row++ {
				cache.ReleaseRow(row)
			}
		}
		data := make([]int32, count*group.length)
		for row, sequence := range group.sequences {
			copy(data[row*group.length:(row+1)*group.length], toI32(sequence.Tokens))
		}
		ids := tensors.NewInt32sWithData([]int{count, group.length}, data)
		combined := tensors.New(count, scheduler.vocab)
		for member, modelWeight := range scheduler.Ensemble.Models {
			logits := modelWeight.Model.PrefillCED(ids, scheduler.cedCaches[member])
			for row, sequence := range group.sequences {
				weight := scheduler.weights[member]
				if index := member; index < len(sequence.weights) {
					weight = sequence.weights[index]
				}
				element := row*group.length + group.length - 1
				addScaledSlice(combined.Data[row*scheduler.vocab:(row+1)*scheduler.vocab],
					logits.Data[element*scheduler.vocab:(element+1)*scheduler.vocab], weight)
			}
		}
		scheduler.stats.PrefillTokens += int64(count * group.length)
		failed := false
		for row, sequence := range group.sequences {
			if failed {
				sequence.Prefilled = 0
				sequence.lastLogits = nil
				continue
			}
			sequence.Prefilled = group.length
			sequence.lastLogits = logitsRow(combined, row, scheduler.vocab)
			if !scheduler.promoteCEDAt(sequence, row) {
				sequence.Prefilled = 0
				sequence.lastLogits = nil
				failed = true
				continue
			}
			*events = append(*events, emitSequence(scheduler.Ensemble.Tokenizer, scheduler.Ensemble.Tools, scheduler.vocab, sequence)...)
			promoted[sequence] = true
		}
		for _, cache := range scheduler.cedCaches {
			for row := 0; row < count; row++ {
				cache.ReleaseRow(row)
			}
		}
	}
}

// promoteCEDAt moves one CED batch row from every member's scratch cache into
// its decode cache and releases the prefill marker row.
func (scheduler *EnsembleScheduler) promoteCEDAt(sequence *Sequence, sourceRow int) bool {
	if sequence == nil || scheduler.cedCaches == nil || sequence.Cache != scheduler.prefillCaches[0] {
		return false
	}
	row := scheduler.freeRow()
	if row < 0 {
		return false
	}
	for member := range scheduler.caches {
		if !scheduler.caches[member].AdoptRowFrom(row, scheduler.cedCaches[member], sourceRow) {
			return false
		}
	}
	for _, cache := range scheduler.cedCaches {
		cache.ReleaseRow(sourceRow)
	}
	prefillRow := sequence.Row
	for _, cache := range scheduler.prefillCaches {
		cache.ReleaseRow(prefillRow)
	}
	if prefillRow >= 0 && prefillRow < len(scheduler.prefillRows) {
		scheduler.prefillRows[prefillRow] = nil
	}
	sequence.Cache = scheduler.caches[0]
	sequence.Row = row
	scheduler.rows[row] = sequence
	sequence.Status = SequenceDecode
	return true
}

// promote moves a completed prefill row into every member's decode cache
// without copying KV blocks, then frees the prefill row.
func (scheduler *EnsembleScheduler) promote(sequence *Sequence) bool {
	if sequence == nil || sequence.Cache != scheduler.prefillCaches[0] {
		return false
	}
	row := scheduler.freeRow()
	if row < 0 {
		return false
	}
	prefillRow := sequence.Row
	for member := range scheduler.caches {
		if !scheduler.caches[member].AdoptRowFrom(row, scheduler.prefillCaches[member], prefillRow) {
			return false
		}
	}
	for _, cache := range scheduler.prefillCaches {
		cache.ReleaseRow(prefillRow)
	}
	if prefillRow >= 0 && prefillRow < len(scheduler.prefillRows) {
		scheduler.prefillRows[prefillRow] = nil
	}
	sequence.Cache = scheduler.caches[0]
	sequence.Row = row
	scheduler.rows[row] = sequence
	sequence.Status = SequenceDecode
	return true
}

func (scheduler *EnsembleScheduler) decodeForward() {
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
	combined := scheduler.forwardMembers(ids, len(positions), 1, positions, scheduler.caches)
	for index, sequence := range positions {
		sequence.lastLogits = logitsRow(combined, index, scheduler.vocab)
	}
}

// forwardMembers runs every member and returns the weighted-sum logits.
func (scheduler *EnsembleScheduler) forwardMembers(ids *tensors.Int32s, rows, chunk int, sequences []*Sequence, caches []*model.KVBuffer) *tensors.Tensor {
	combined := tensors.New(rows*chunk, scheduler.vocab)
	for index, member := range scheduler.Ensemble.Models {
		logits := member.Model.Forward(ids, caches[index])
		for row := 0; row < rows; row++ {
			weight := scheduler.weights[index]
			if row < len(sequences) && index < len(sequences[row].weights) {
				weight = sequences[row].weights[index]
			}
			for token := 0; token < chunk; token++ {
				element := row*chunk + token
				addScaledSlice(combined.Data[element*scheduler.vocab:(element+1)*scheduler.vocab],
					logits.Data[element*scheduler.vocab:(element+1)*scheduler.vocab], weight)
			}
		}
	}
	return combined
}

func normalizeWeightValues(values []float32) []float32 {
	weights := append([]float32(nil), values...)
	var sum float32
	for index, value := range weights {
		if value < 0 {
			value = 0
			weights[index] = 0
		}
		sum += value
	}
	if sum <= 0 {
		for index := range weights {
			weights[index] = 1 / float32(len(weights))
		}
		return weights
	}
	for index := range weights {
		weights[index] /= sum
	}
	return weights
}

// releaseSequence frees whichever member caches own the sequence row and
// returns its per-member block reservations.
func (scheduler *EnsembleScheduler) releaseSequence(sequence *Sequence) {
	switch {
	case sequence.Cache == nil:
	case len(scheduler.caches) > 0 && sequence.Cache == scheduler.caches[0]:
		for _, cache := range scheduler.caches {
			cache.ReleaseRow(sequence.Row)
		}
		if sequence.Row >= 0 && sequence.Row < len(scheduler.rows) {
			scheduler.rows[sequence.Row] = nil
		}
	case len(scheduler.prefillCaches) > 0 && sequence.Cache == scheduler.prefillCaches[0]:
		for _, cache := range scheduler.prefillCaches {
			cache.ReleaseRow(sequence.Row)
		}
		if sequence.Row >= 0 && sequence.Row < len(scheduler.prefillRows) {
			scheduler.prefillRows[sequence.Row] = nil
		}
	default:
		sequence.Cache.Release()
	}
	if len(sequence.memberReserved) == len(scheduler.reserved) {
		for member, blocks := range sequence.memberReserved {
			scheduler.reserved[member] -= blocks
			if scheduler.reserved[member] < 0 {
				scheduler.reserved[member] = 0
			}
		}
	}
	sequence.memberReserved = nil
	sequence.Cache = nil
	sequence.Row = -1
	sequence.Solo = false
}

func (scheduler *EnsembleScheduler) preempt(sequence *Sequence) {
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

func (scheduler *EnsembleScheduler) reap() {
	active := scheduler.running[:0]
	for _, sequence := range scheduler.running {
		if sequence.Status != SequenceFinished && sequence.Status != SequenceFailed {
			active = append(active, sequence)
			continue
		}
		scheduler.releaseSequence(sequence)
		scheduler.finished = append(scheduler.finished, sequence)
	}
	scheduler.running = active
}

func (scheduler *EnsembleScheduler) compactRows() {
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
		moved := true
		for _, cache := range scheduler.caches {
			if !cache.MoveRow(free, high) {
				moved = false
				break
			}
		}
		if !moved {
			break
		}
		scheduler.rows[free] = sequence
		scheduler.rows[high] = nil
		sequence.Row = free
	}
}

func (scheduler *EnsembleScheduler) compactPrefillRows() {
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
		moved := true
		for _, cache := range scheduler.prefillCaches {
			if !cache.MoveRow(free, high) {
				moved = false
				break
			}
		}
		if !moved {
			break
		}
		scheduler.prefillRows[free] = sequence
		scheduler.prefillRows[high] = nil
		sequence.Row = free
	}
}

// Cancel removes one request and releases its rows.
func (scheduler *EnsembleScheduler) Cancel(sequence *Sequence) {
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

// SortFinishedEnsemble sorts finished sequences by ID for deterministic tests.
func SortFinishedEnsemble(sequences []*Sequence) {
	sort.SliceStable(sequences, func(left, right int) bool { return sequences[left].ID < sequences[right].ID })
}
