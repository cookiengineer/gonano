package data

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/cookiengineer/gonano/tokenizer"
)

// conversationLine is one JSONL row. Two shapes are accepted:
//
//   - a chat transcript: {"messages":[{"role","content","thinking"|"reasoning_content"}]}
//   - a reasoning record: {"instruction","response","thinking","intent",...}
//
// The reasoning shape is what gonano-school's `cmd/reasoning` emits from the
// DeepSeek-R1 trace datasets; when an `intent` is present it is prepended to
// the reasoning trace so the model learns the "The user wants ..." style.
type conversationLine struct {
	System      string                `json:"system"`
	Instruction string                `json:"instruction"`
	Response    string                `json:"response"`
	Thinking    string                `json:"thinking"`
	Intent      string                `json:"intent"`
	Messages    []conversationMessage `json:"messages"`
}

type conversationMessage struct {
	Role             string `json:"role"`
	Content          string `json:"content"`
	Thinking         string `json:"thinking"`
	ReasoningContent string `json:"reasoning_content"`
}

// ConversationSource streams tokenizer.Conversations from one JSONL file or a
// directory of JSONL files (read in sorted order). It implements the same
// Next() shape as data.ConvProvider.
type ConversationSource struct {
	files     []string
	index     int
	file      *os.File
	reader    *bufio.Reader
	batchSize int
	err       error
	closed    bool
}

// OpenConversations opens a JSONL file or directory of JSONL files for SFT.
func OpenConversations(path string, batchSize int) (*ConversationSource, error) {
	if batchSize <= 0 {
		batchSize = 1
	}
	info, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("conversations: %w", err)
	}
	files := []string{path}
	if info.IsDir() {
		entries, err := os.ReadDir(path)
		if err != nil {
			return nil, fmt.Errorf("conversations: %w", err)
		}
		files = nil
		for _, entry := range entries {
			if !entry.IsDir() && strings.HasSuffix(entry.Name(), ".jsonl") {
				files = append(files, filepath.Join(path, entry.Name()))
			}
		}
		sort.Strings(files)
	}
	if len(files) == 0 {
		return nil, fmt.Errorf("conversations: no .jsonl files in %s", path)
	}
	source := &ConversationSource{files: files, batchSize: batchSize}
	source.openNext()
	if source.err != nil {
		return nil, source.err
	}
	return source, nil
}

// Next returns up to batchSize conversations. ok is false once every file is
// exhausted. Parse errors are skipped line by line; I/O errors are reported by
// Err.
func (source *ConversationSource) Next() ([]*tokenizer.Conversation, bool) {
	if source.err != nil || source.closed {
		return nil, false
	}
	batch := make([]*tokenizer.Conversation, 0, source.batchSize)
	for len(batch) < source.batchSize {
		line, ok := source.nextLine()
		if !ok {
			break
		}
		if conversation, ok := parseConversationLine(line); ok {
			batch = append(batch, conversation)
		}
	}
	if len(batch) == 0 {
		return nil, false
	}
	return batch, true
}

// Err reports the first I/O error encountered.
func (source *ConversationSource) Err() error { return source.err }

// Close releases the open file.
func (source *ConversationSource) Close() error {
	source.closed = true
	if source.file != nil {
		err := source.file.Close()
		source.file = nil
		return err
	}
	return nil
}

func (source *ConversationSource) openNext() bool {
	for source.index < len(source.files) {
		path := source.files[source.index]
		source.index++
		file, err := os.Open(path)
		if err != nil {
			source.err = err
			return false
		}
		source.file = file
		source.reader = bufio.NewReaderSize(file, 1<<20)
		return true
	}
	return false
}

func (source *ConversationSource) nextLine() ([]byte, bool) {
	for {
		if source.reader == nil {
			if !source.openNext() {
				return nil, false
			}
		}
		line, err := source.reader.ReadBytes('\n')
		if len(line) > 0 {
			return bytes.TrimSpace(line), true
		}
		if err != nil {
			source.file.Close()
			source.file = nil
			source.reader = nil
			if err == io.EOF {
				continue
			}
			source.err = err
			return nil, false
		}
	}
}

func parseConversationLine(data []byte) (*tokenizer.Conversation, bool) {
	if len(data) == 0 {
		return nil, false
	}
	var line conversationLine
	if err := json.Unmarshal(data, &line); err != nil {
		return nil, false
	}
	conversation := &tokenizer.Conversation{}
	if len(line.Messages) > 0 {
		for _, message := range line.Messages {
			thinking := strings.TrimSpace(message.Thinking)
			if thinking == "" {
				thinking = strings.TrimSpace(message.ReasoningContent)
			}
			conversation.Messages = append(conversation.Messages, tokenizer.Message{
				Role:     strings.TrimSpace(message.Role),
				Content:  message.Content,
				Thinking: thinking,
			})
		}
	} else {
		instruction := strings.TrimSpace(line.Instruction)
		if instruction == "" {
			return nil, false
		}
		if system := strings.TrimSpace(line.System); system != "" {
			conversation.Messages = append(conversation.Messages, tokenizer.Message{Role: "system", Content: system})
		}
		conversation.Messages = append(conversation.Messages, tokenizer.Message{Role: "user", Content: instruction})
		thinking := strings.TrimSpace(line.Thinking)
		if thinking != "" {
			if intent := strings.TrimSpace(line.Intent); intent != "" {
				thinking = intent + "\n\n" + thinking
			}
		}
		conversation.Messages = append(conversation.Messages, tokenizer.Message{Role: "assistant", Content: line.Response, Thinking: thinking})
	}
	if len(conversation.Messages) == 0 {
		return nil, false
	}
	for _, message := range conversation.Messages {
		if message.Thinking != "" {
			conversation.Extra = map[string]any{"thinking": true}
			break
		}
	}
	return conversation, true
}
