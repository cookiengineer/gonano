package data

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

// BaseDir returns the directory where gonano caches datasets, tokenizers, and
// checkpoints. It honors GONANO_BASE_DIR, defaulting to ~/.cache/gonano.
func BaseDir() string {
	if dir := os.Getenv("GONANO_BASE_DIR"); dir != "" {
		os.MkdirAll(dir, 0o755)
		return dir
	}
	home, err := os.UserHomeDir()
	if err != nil {
		home = "."
	}
	dir := filepath.Join(home, ".cache", "gonano")
	os.MkdirAll(dir, 0o755)
	return dir
}

// DownloadFile downloads url to path, writing to a temp file and renaming on
// success. It retries with exponential backoff. If path already exists, it
// returns immediately.
func DownloadFile(url, path string) error {
	if _, err := os.Stat(path); err == nil {
		return nil
	}
	const maxAttempts = 5
	var lastErr error
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		if err := downloadOnce(url, path); err == nil {
			return nil
		} else {
			lastErr = err
		}
		if attempt < maxAttempts {
			time.Sleep(time.Duration(1<<attempt) * time.Second)
		}
	}
	return fmt.Errorf("download %s: %w", url, lastErr)
}

func downloadOnce(url, path string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("status %d", resp.StatusCode)
	}
	tmp := path + ".tmp"
	file, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if _, err := io.Copy(file, resp.Body); err != nil {
		file.Close()
		os.Remove(tmp)
		return err
	}
	if err := file.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		os.Remove(tmp)
		return err
	}
	return nil
}

// DownloadWithLock downloads url to path under a .lock file so that concurrent
// processes download it only once. It returns the local path.
func DownloadWithLock(url, path string) (string, error) {
	if _, err := os.Stat(path); err == nil {
		return path, nil
	}
	lockPath := path + ".lock"
	for {
		lock, err := os.OpenFile(lockPath, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
		if err == nil {
			lock.Close()
			defer os.Remove(lockPath)
			if _, err := os.Stat(path); err == nil {
				return path, nil
			}
			if err := DownloadFile(url, path); err != nil {
				return path, err
			}
			return path, nil
		}
		// Another process holds the lock; wait and recheck.
		time.Sleep(200 * time.Millisecond)
		if _, err := os.Stat(path); err == nil {
			return path, nil
		}
	}
}

// ListHFParquetShards returns the parquet shard URLs for a HuggingFace Hub
// dataset, using the auto-generated parquet export API.
func ListHFParquetShards(repoID, config, split string) ([]string, error) {
	url := fmt.Sprintf("https://huggingface.co/api/datasets/%s/parquet/%s/%s", repoID, config, split)
	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("list shards: status %d", resp.StatusCode)
	}
	var shards []string
	if err := json.NewDecoder(resp.Body).Decode(&shards); err != nil {
		return nil, err
	}
	return shards, nil
}

// HFRow is a single row returned by the datasets-server API.
type HFRow struct {
	Row    map[string]json.RawMessage `json:"row"`
	RowIdx int                        `json:"row_idx"`
}

type hfRowsResponse struct {
	Rows         []HFRow `json:"rows"`
	NumRowsTotal int64   `json:"num_rows_total"`
}

// FetchHFRows fetches rows of a HuggingFace dataset via the datasets-server
// REST API, returning them as raw JSON objects. This is used for small task
// datasets where nested parquet decoding is not required.
func FetchHFRows(repoID, config, split string, offset, length int) ([]map[string]json.RawMessage, int64, error) {
	url := fmt.Sprintf("https://datasets-server.huggingface.co/rows?dataset=%s&config=%s&split=%s&offset=%d&length=%d",
		repoID, config, split, offset, length)
	resp, err := http.Get(url)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, 0, fmt.Errorf("fetch rows: status %d", resp.StatusCode)
	}
	var body hfRowsResponse
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return nil, 0, err
	}
	rows := make([]map[string]json.RawMessage, len(body.Rows))
	for index, hfRow := range body.Rows {
		rows[index] = hfRow.Row
	}
	return rows, body.NumRowsTotal, nil
}
