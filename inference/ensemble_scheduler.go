package inference

import (
	"sort"

	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tensors"
)

// EnsembleScheduler batches independent requests that share one Ensemble. It
// keeps one paged work cache per member and advances the same rows through every
// member, sampling once from the weighted-sum logits.
type EnsembleScheduler struct {
	Ensemble     *Ensemble
	MaxActive    int
	PrefillChunk int

	waiting  []*Sequence
	running  []*Sequence
	finished []*Sequence

	rows     []*Sequence
	caches   []*model.KVBuffer
	weights  []float32
	sequence int
	vocab    int
	nextSeed uint64
	stats    SchedulerStats
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
	for _, cache := range scheduler.caches {
		if manager := cache.BlockManager(); manager != nil {
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
	if len(scheduler.caches) > 0 {
		return
	}
	maxRows := scheduler.MaxActive
	if maxRows <= 0 {
		maxRows = 8
		scheduler.MaxActive = maxRows
	}
	scheduler.vocab = scheduler.Ensemble.Models[0].Model.Config.VocabSize
	scheduler.sequence = scheduler.Ensemble.Models[0].Model.Config.SequenceLen
	for _, member := range scheduler.Ensemble.Models {
		if member.Model.Config.VocabSize != scheduler.vocab {
			panic("inference: ensemble models must share a vocab size")
		}
		if member.Model.Config.SequenceLen < scheduler.sequence {
			scheduler.sequence = member.Model.Config.SequenceLen
		}
	}
	scheduler.weights = scheduler.Ensemble.normalizedWeights()
	scheduler.caches = make([]*model.KVBuffer, len(scheduler.Ensemble.Models))
	for index, member := range scheduler.Ensemble.Models {
		scheduler.caches[index] = newCacheForModel(member.Model.Config, maxRows, member.Model.Config.SequenceLen)
	}
	scheduler.rows = make([]*Sequence, maxRows)
}

func (scheduler *EnsembleScheduler) freeRow() int {
	for index, sequence := range scheduler.rows {
		if sequence == nil {
			return index
		}
	}
	return -1
}

func (scheduler *EnsembleScheduler) admit() {
	limit := scheduler.MaxActive
	for len(scheduler.waiting) > 0 && (limit <= 0 || len(scheduler.running) < limit) {
		row := scheduler.freeRow()
		if row < 0 {
			break
		}
		sequence := scheduler.waiting[0]
		scheduler.waiting = scheduler.waiting[1:]
		for _, cache := range scheduler.caches {
			cache.SetPositionAt(row, 0)
		}
		sequence.Row = row
		sequence.Cache = scheduler.caches[0]
		sequence.Status = SequencePrefill
		scheduler.rows[row] = sequence
		scheduler.running = append(scheduler.running, sequence)
	}
}

// Step advances prefills and decodes in one batched pass over every member.
func (scheduler *EnsembleScheduler) Step() []Event {
	scheduler.ensureCaches()
	scheduler.admit()
	events := []Event{}
	sampled := make(map[*Sequence]bool)

	hasDecode := false
	for _, sequence := range scheduler.running {
		if sequence.Status == SequenceDecode {
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
		events = append(events, emitSequence(scheduler.Ensemble.Tokenizer, scheduler.Ensemble.Tools, scheduler.vocab, sequence)...)
	}
	scheduler.reap()
	scheduler.compactRows()
	if hasDecode || len(sampled) > 0 {
		scheduler.mixedForward()
	}
	scheduler.reap()
	scheduler.compactRows()
	for _, event := range events {
		if event.TokenMask == 1 {
			scheduler.stats.Generated++
		}
	}
	return events
}

func (scheduler *EnsembleScheduler) chunkedPrefill() ([]Event, map[*Sequence]bool) {
	events := []Event{}
	promoted := make(map[*Sequence]bool)
	active := make([]*Sequence, 0, len(scheduler.rows))
	for _, sequence := range scheduler.rows {
		if sequence != nil && sequence.Status == SequencePrefill {
			active = append(active, sequence)
		}
	}
	if len(active) == 0 {
		return events, promoted
	}
	chunk := 0
	for _, sequence := range active {
		remaining := len(sequence.Prompt) - sequence.Prefilled
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
	data := make([]int32, len(active)*chunk)
	for index, sequence := range active {
		copy(data[index*chunk:(index+1)*chunk], toI32(sequence.Prompt[sequence.Prefilled:sequence.Prefilled+chunk]))
	}
	scheduler.stats.PrefillTokens += int64(len(active) * chunk)
	ids := tensors.NewInt32sWithData([]int{len(active), chunk}, data)
	combined := scheduler.forwardMembers(ids, len(active), chunk, active)
	for index, sequence := range active {
		sequence.Prefilled += chunk
		sequence.lastLogits = logitsRow(combined, index*chunk+chunk-1, scheduler.vocab)
		if sequence.Prefilled == len(sequence.Prompt) {
			sequence.Status = SequenceDecode
			events = append(events, emitSequence(scheduler.Ensemble.Tokenizer, scheduler.Ensemble.Tools, scheduler.vocab, sequence)...)
			promoted[sequence] = true
		}
	}
	return events, promoted
}

func (scheduler *EnsembleScheduler) mixedForward() {
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
			if sequence.Prefilled >= len(sequence.Prompt) {
				sequence.Status = SequenceDecode
				continue
			}
			tokens[count] = int32(sequence.Prompt[sequence.Prefilled])
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
	combined := scheduler.forwardMembers(ids, count, 1, positions)
	for index := 0; index < count; index++ {
		sequence := positions[index]
		sequence.lastLogits = logitsRow(combined, index, scheduler.vocab)
		if sequence.Status != SequencePrefill {
			continue
		}
		sequence.Prefilled++
		if sequence.Prefilled == len(sequence.Prompt) {
			sequence.Status = SequenceDecode
		}
	}
}

// forwardMembers runs every member and returns the weighted-sum logits.
func (scheduler *EnsembleScheduler) forwardMembers(ids *tensors.Int32s, rows, chunk int, sequences []*Sequence) *tensors.Tensor {
	combined := tensors.New(rows*chunk, scheduler.vocab)
	for index, member := range scheduler.Ensemble.Models {
		logits := member.Model.Forward(ids, scheduler.caches[index])
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

func (scheduler *EnsembleScheduler) reap() {
	active := scheduler.running[:0]
	for _, sequence := range scheduler.running {
		if sequence.Status != SequenceFinished && sequence.Status != SequenceFailed {
			active = append(active, sequence)
			continue
		}
		for _, cache := range scheduler.caches {
			cache.ReleaseRow(sequence.Row)
		}
		if sequence.Row >= 0 && sequence.Row < len(scheduler.rows) {
			scheduler.rows[sequence.Row] = nil
		}
		sequence.Row = -1
		sequence.Cache = nil
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
		for _, cache := range scheduler.caches {
			cache.ReleaseRow(sequence.Row)
		}
		if sequence.Row >= 0 && sequence.Row < len(scheduler.rows) {
			scheduler.rows[sequence.Row] = nil
		}
		sequence.Row = -1
		sequence.Cache = nil
		sequence.Status = SequenceFailed
		scheduler.running = append(scheduler.running[:index], scheduler.running[index+1:]...)
		scheduler.compactRows()
		return
	}
}

// SortFinishedEnsemble sorts finished sequences by ID for deterministic tests.
func SortFinishedEnsemble(sequences []*Sequence) {
	sort.SliceStable(sequences, func(left, right int) bool { return sequences[left].ID < sequences[right].ID })
}
