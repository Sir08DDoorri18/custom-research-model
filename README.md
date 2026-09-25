# custom-research-model

A research partner for math, science and engineering that runs on localhost. It chats
normally, looks things up only when a question needs facts, and checks every cited sentence
of its answer against the source it cites, using several independent models.

```
browser  localhost:8000  (Chainlit: chat + every step, model verdict and reasoning, live)
   │
brain    Claude via the local `claude` CLI (Haiku / Sonnet / Opus)  or  free models only
   │     decides when to search, which sources to read, and writes the answer
   ▼
tools    search_papers   OpenAlex · Semantic Scholar · arXiv
         search_web      DuckDuckGo
         citation_graph  Semantic Scholar references / citations
         read_sources    fetch → BM25 → cheap model reads passages (relevance, summary,
                         verbatim quote) → quote checked against the text → evidence E#
   ▼
check    every sentence citing E#:
         ① quote string match  ② local NLI  ③ first judge (Groq)
         ④ jury of 3 other model families, only where ② and ③ disagree
```

Every step is also written to `traces/<time>-<session>.md` with prompts, reasoning and outputs.
Diagrams of each flow, what the on-screen marks mean and a troubleshooting table (Korean):
[docs/FLOW.md](docs/FLOW.md).

## Adding files

Drop PDFs, Word/PowerPoint/Excel files, csv/txt/md or pictures into the chat box (or
`ask --file`). Text, tables and equations (as LaTeX) are read on this PC with Docling; each
figure inside a document and each photo is described by the `vision` model, so those images
are sent to that provider. With a Claude preset, attached photos are also shown to Claude
directly. Files become citable sources (`내 파일`, with page numbers; `그림해석` for figures,
whose numbers are approximate). The first read of a 15-page paper takes a few minutes on a
laptop CPU; results are cached in `.cache/files`.

Docling runs in a child process whose home/temp paths point to `.cache/tmp`, because its PDF
engine fails on non-ASCII Windows user folders.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -r requirements-nli.txt   # optional: local NLI check
copy .env.example .env            # then paste your API keys into .env
python -m research doctor         # checks keys, models and search APIs
python -m research serve          # opens http://localhost:8000
```

- **Claude presets** need [Claude Code](https://claude.com/claude-code) installed and logged in
  (`claude` on PATH). They run on your own subscription on your own machine; do not expose
  this app to other people with your login. Anyone else should use their own login or an API key.
- **Free models** (all optional, no card needed): Groq, Mistral (Experiment plan), OpenRouter,
  NVIDIA NIM. A missing key only switches off the steps that need it.
- The free tiers of Mistral and Google may use your prompts for training.

## Commands

| command | what it does |
|---|---|
| `python -m research serve [--port 8000] [--headless]` | chat UI |
| `python -m research ask "question" [--preset saver] [--file paper.pdf ...]` | one question in the terminal, every step printed |
| `python -m research doctor [--claude]` | which keys, models and APIs work right now |
| `python -m research clean [--all]` | trim caches and old logs now (also runs when the UI starts; limits under `storage` in models.yaml) |
| `python -m research eval` | accuracy of each judge on `eval/claims.jsonl` and how correlated their mistakes are |

## Changing models

Everything lives in [models.yaml](models.yaml): presets, the ordered fallback list for each
role, the jury, rate-limit spacing per provider. Providers retire and rename free models
often; when `doctor` shows `x`, replace that line and run `eval` to see whether the new
judge is any good.

## Limits

- Papers without an open-access PDF are read from the abstract only (marked `초록`).
- The checker verifies that a sentence matches the passage it cites. It cannot tell whether
  the source itself is wrong; retracted papers are flagged when OpenAlex knows about them.
- Semantic Scholar's anonymous quota is shared and often returns 429; a free key
  (`S2_API_KEY`) fixes that.
