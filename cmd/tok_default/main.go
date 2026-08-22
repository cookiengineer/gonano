// Command tok_default regenerates the bundled default tokenizers in
// tokenizer/defaults:
//
//   - byte.json      a byte-level tokenizer (256 single-byte ranks)
//   - markdown.json  a BPE tokenizer trained on representative Markdown
//
// Run it from the repository root after changing the corpus:
//
//	go run ./cmd/tok_default
package main

import (
	"flag"
	"log/slog"
	"os"
	"path/filepath"

	"github.com/cookiengineer/gonano/logging"
	"github.com/cookiengineer/gonano/tokenizer"
)

// markdownVocabSize is the target vocabulary for the Markdown default.
const markdownVocabSize = 4096

func main() {
	out := flag.String("out", "tokenizer/defaults", "output directory")
	flag.Parse()

	logger := logging.Default(slog.LevelInfo)
	os.MkdirAll(*out, 0o755)

	// Byte-level default.
	byteTok := byteTokenizer()
	if err := byteTok.Save(filepath.Join(*out, "byte.json")); err != nil {
		logger.Error("save byte.json", "err", err)
		os.Exit(1)
	}

	// Markdown default: train BPE over representative Markdown.
	var pieces []string
	for _, doc := range markdownCorpus {
		pieces = append(pieces, tokenizer.SplitPieces(doc)...)
	}
	numMerges := markdownVocabSize - 256 - len(tokenizer.SpecialTokens)
	ranks := tokenizer.TrainBPE(pieces, numMerges)
	mdTok := tokenizer.NewTokenizer(ranks, tokenizer.SpecialTokens)
	if err := mdTok.Save(filepath.Join(*out, "markdown.json")); err != nil {
		logger.Error("save markdown.json", "err", err)
		os.Exit(1)
	}

	logger.Info("wrote default tokenizers",
		"byte_vocab", byteTok.VocabSize(),
		"markdown_vocab", mdTok.VocabSize(),
		"dir", *out,
	)
}

func byteTokenizer() *tokenizer.Tokenizer {
	ranks := make(map[string]int, 256)
	for i := 0; i < 256; i++ {
		ranks[string([]byte{byte(i)})] = i
	}
	return tokenizer.NewTokenizer(ranks, tokenizer.SpecialTokens)
}

// markdownCorpus is a sample of Markdown covering the constructs commonly
// found in webdata that has been converted to Markdown: headings, emphasis,
// lists, links, images, code, tables, blockquotes, and ordinary prose.
var markdownCorpus = []string{
	"# Introduction to Machine Learning",
	"Machine learning is the study of computer algorithms that improve automatically through experience.",
	"## Supervised Learning",
	"In supervised learning, a model learns a mapping from inputs to outputs given labeled examples.",
	"A training set consists of input-output pairs, and the model is fit to minimize a loss function.",
	"## Unsupervised Learning",
	"Unsupervised learning finds structure in unlabeled data, such as clusters or low-dimensional representations.",
	"### Clustering",
	"Clustering groups similar examples together without using any labels.",
	"Common algorithms include k-means, hierarchical clustering, and density-based methods.",
	"### Dimensionality Reduction",
	"Principal component analysis projects high-dimensional data onto a lower-dimensional subspace.",
	"## Reinforcement Learning",
	"An agent learns a policy by interacting with an environment and receiving rewards.",
	"The agent aims to maximize the cumulative reward over time.",
	"## Common Notation",
	"A model with parameters $\\theta$ is denoted $f(x; \\theta)$, where $x$ is the input.",
	"Gradients are computed with respect to the parameters using backpropagation.",
	"## Further Reading",
	"See the [introduction to deep learning](https://example.com/deep-learning) for more details.",
	"",
	"# Getting Started",
	"First, install the dependencies with your package manager:",
	"```bash",
	"sudo apt update && sudo apt install build-essential",
	"```",
	"Then clone the repository and build it:",
	"```bash",
	"git clone https://github.com/example/project.git",
	"cd project",
	"make build",
	"```",
	"",
	"# Features",
	"- **Fast** — optimized for speed.",
	"- **Portable** — runs on Linux, macOS, and Windows.",
	"- *Lightweight* — a single binary with no runtime dependencies.",
	"- **Extensible** — a clean plugin interface.",
	"",
	"1. Install the binary.",
	"2. Configure the server.",
	"3. Start the service.",
	"4. Verify the logs.",
	"",
	"# Configuration",
	"The configuration file uses YAML syntax:",
	"```yaml",
	"server:",
	"  host: 0.0.0.0",
	"  port: 8080",
	"logging:",
	"  level: info",
	"  format: json",
	"```",
	"",
	"| Option | Type | Default | Description |",
	"| --- | --- | --- | --- |",
	"| host | string | localhost | The bind address |",
	"| port | int | 8080 | The listen port |",
	"| verbose | bool | false | Enable debug output |",
	"",
	"# Frequently Asked Questions",
	"> Why does the build fail?",
	"> Make sure all dependencies are installed and the environment is configured correctly.",
	"",
	"> Can I use this in production?",
	"> Yes, it is designed for production workloads.",
	"",
	"# API Reference",
	"## GET /users",
	"Returns a list of users. The response is JSON:",
	"```json",
	"{",
	"  \"users\": [",
	"    {\"id\": 1, \"name\": \"Alice\"},",
	"    {\"id\": 2, \"name\": \"Bob\"}",
	"  ]",
	"}",
	"```",
	"",
	"## POST /users",
	"Creates a new user. The request body must contain a `name` field.",
	"Authentication is performed with a bearer token in the `Authorization` header.",
	"",
	"# Release Notes",
	"## Version 1.0.0",
	"- Initial public release.",
	"- Added support for Windows and macOS.",
	"- Fixed a memory leak in the parser.",
	"",
	"## Version 0.9.0",
	"- Beta release.",
	"- Improved error messages.",
	"",
	"# The History of Science",
	"Science is the systematic study of the natural world through observation and experiment.",
	"The scientific method involves forming hypotheses, conducting experiments, and analyzing results.",
	"Over centuries, science has transformed medicine, communication, and transportation.",
	"",
	"# Programming Languages",
	"Go is a statically typed, compiled programming language designed at Google.",
	"Python is a dynamically typed, interpreted language known for its readability.",
	"Rust emphasizes memory safety without garbage collection.",
	"JavaScript is the language of the web browser and the server.",
	"",
	"# Healthy Eating",
	"A balanced diet includes fruits, vegetables, whole grains, and lean proteins.",
	"Regular exercise is also important for maintaining good health.",
	"",
	"# Travel Guide",
	"Paris is the capital of France and is famous for the Eiffel Tower.",
	"The Louvre museum houses thousands of works of art.",
	"",
	"# Notes",
	"This is **bold text** and this is *italic text*.",
	"This is ***bold and italic*** text.",
	"This is `inline code` within a sentence.",
	"This is ~~strikethrough~~ text.",
	"A link to [Wikipedia](https://en.wikipedia.org) appears here.",
	"An image: ![alt text](https://example.com/image.png \"title\")",
	"",
	"# Conclusion",
	"Thank you for reading this document. We hope you found it helpful.",
	"Please report any issues on the project page.",
}
