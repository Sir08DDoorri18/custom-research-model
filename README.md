# custom-research-model

Localhost research assistant for math, science and engineering. It searches papers and the web
only when needed, and checks every cited sentence of its answer against the source with
several independent models. In parallel mode, several models research the question at once
from different angles (papers, official sources, news, counter-evidence) and their findings
are compared side by side. Conversations are saved on this machine (SQLite, deleted after 30
days) and can be reopened after a reload. Flow diagrams and troubleshooting: [docs/FLOW.md](docs/FLOW.md).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -r requirements-nli.txt   # optional
copy .env.example .env                # add API keys
python -m research doctor
python -m research serve              # http://localhost:8000
```

Claude presets use your own logged-in [Claude Code](https://claude.com/claude-code) on your own
machine. Free-model keys (Groq, Mistral, OpenRouter, NVIDIA) are optional.

## Commands

| command | |
|---|---|
| `serve` | chat UI |
| `ask "question" [--file x.pdf] [--mode parallel]` | one question in the terminal |
| `doctor` | check keys and models |
| `clean [--all]` | trim caches |
| `eval` | score the judge models |

Run as `python -m research <command>`. Settings live in [models.yaml](models.yaml).

MIT licensed. Dependencies keep their own licenses (pymupdf4llm is AGPL). Tested on Windows, Python 3.14.
