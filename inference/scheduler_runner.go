package inference

import (
	"sync"
	"time"
)

// SchedulerSubmission is one request submitted to a SchedulerRunner. Events are
// queued without blocking the scheduler and consumed with Next.
type SchedulerSubmission struct {
	request Request
	runner  *SchedulerRunner

	mutex    sync.Mutex
	cond     *sync.Cond
	events   []Event
	head     int
	closed   bool
	canceled bool
	sequence *Sequence
	closeOne sync.Once
}

// Next returns the next event, blocking until one is available or the
// submission is closed.
func (submission *SchedulerSubmission) Next() (Event, bool) {
	submission.mutex.Lock()
	defer submission.mutex.Unlock()
	for submission.head >= len(submission.events) && !submission.closed {
		submission.cond.Wait()
	}
	if submission.head >= len(submission.events) {
		return Event{}, false
	}
	event := submission.events[submission.head]
	submission.head++
	if submission.head > 4096 && submission.head*2 > len(submission.events) {
		submission.events = append([]Event(nil), submission.events[submission.head:]...)
		submission.head = 0
	}
	return event, true
}

// Cancel asks the runner to drop the request. It is safe to call after the
// submission has already finished.
func (submission *SchedulerSubmission) Cancel() {
	submission.mutex.Lock()
	if submission.canceled || submission.closed {
		submission.mutex.Unlock()
		return
	}
	submission.canceled = true
	submission.mutex.Unlock()
	select {
	case submission.runner.cancel <- submission:
	case <-submission.runner.stop:
	}
}

func (submission *SchedulerSubmission) push(event Event) {
	submission.mutex.Lock()
	if !submission.closed {
		submission.events = append(submission.events, event)
		submission.cond.Signal()
	}
	submission.mutex.Unlock()
}

func (submission *SchedulerSubmission) close() {
	submission.closeOne.Do(func() {
		submission.mutex.Lock()
		submission.closed = true
		submission.cond.Broadcast()
		submission.mutex.Unlock()
	})
}

// schedulerDriver is the shared contract of the single-model and ensemble
// schedulers.
type schedulerDriver interface {
	Submit(Request) *Sequence
	Step() []Event
	Cancel(*Sequence)
	Pending() bool
}

// SchedulerRunner drives a scheduler in the background and dispatches events to
// concurrent submissions. It is the bridge used by the HTTP server.
type SchedulerRunner struct {
	driver schedulerDriver
	submit chan *SchedulerSubmission
	cancel chan *SchedulerSubmission
	stop   chan struct{}
	done   chan struct{}

	startOnce sync.Once
	stopOnce  sync.Once

	requests map[*Sequence]*SchedulerSubmission
}

// NewSchedulerRunner builds a runner over scheduler.
func NewSchedulerRunner(scheduler *Scheduler) *SchedulerRunner {
	return NewEnsembleSchedulerRunner(scheduler)
}

// NewEnsembleSchedulerRunner builds a runner over any scheduler driver.
func NewEnsembleSchedulerRunner(scheduler schedulerDriver) *SchedulerRunner {
	return &SchedulerRunner{
		driver:   scheduler,
		submit:   make(chan *SchedulerSubmission, 64),
		cancel:   make(chan *SchedulerSubmission, 64),
		stop:     make(chan struct{}),
		done:     make(chan struct{}),
		requests: make(map[*Sequence]*SchedulerSubmission),
	}
}

// Scheduler returns the underlying single-model scheduler, if any.
func (runner *SchedulerRunner) Scheduler() *Scheduler {
	scheduler, _ := runner.driver.(*Scheduler)
	return scheduler
}

// Start begins the background loop. It is safe to call more than once.
func (runner *SchedulerRunner) Start() {
	runner.startOnce.Do(func() { go runner.run() })
}

// Stop halts the runner and closes every outstanding submission.
func (runner *SchedulerRunner) Stop() {
	runner.stopOnce.Do(func() {
		close(runner.stop)
		<-runner.done
	})
}

// Submit queues a request and starts the runner if needed.
func (runner *SchedulerRunner) Submit(request Request) *SchedulerSubmission {
	submission := &SchedulerSubmission{request: request, runner: runner}
	submission.cond = sync.NewCond(&submission.mutex)
	runner.Start()
	select {
	case runner.submit <- submission:
	case <-runner.stop:
		submission.close()
	}
	return submission
}

func (runner *SchedulerRunner) run() {
	defer close(runner.done)
	ticker := time.NewTicker(time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case submission := <-runner.submit:
			submission.mutex.Lock()
			canceled := submission.canceled
			submission.mutex.Unlock()
			if canceled {
				submission.close()
				continue
			}
			submission.sequence = runner.driver.Submit(submission.request)
			runner.requests[submission.sequence] = submission
			runner.step()
		case submission := <-runner.cancel:
			runner.cancelSubmission(submission)
		case <-ticker.C:
			if runner.driver.Pending() {
				runner.step()
			}
		case <-runner.stop:
			for sequence, submission := range runner.requests {
				runner.driver.Cancel(sequence)
				submission.close()
				delete(runner.requests, sequence)
			}
			return
		}
	}
}

func (runner *SchedulerRunner) step() {
	for _, event := range runner.driver.Step() {
		submission := runner.requests[event.Sequence]
		if submission == nil {
			continue
		}
		submission.push(event)
		if event.Finished {
			submission.close()
			delete(runner.requests, event.Sequence)
		}
	}
}

func (runner *SchedulerRunner) cancelSubmission(submission *SchedulerSubmission) {
	if submission.sequence != nil {
		if _, exists := runner.requests[submission.sequence]; exists {
			runner.driver.Cancel(submission.sequence)
			delete(runner.requests, submission.sequence)
		}
	}
	submission.close()
}
