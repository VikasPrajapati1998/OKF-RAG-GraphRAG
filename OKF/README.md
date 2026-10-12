# OKF Chatbot

Command-line chatbot that answers questions about a document using the
**Open Knowledge Format (OKF)**: the document is converted into structured,
self-describing knowledge records, and questions are answered by retrieving
those records instead of raw text chunks. Runs fully locally with Ollama; no UI.

## OKF record format

Each chunk becomes one record (`okf_version` `1.0`):

```json
{
  "okf_version": "1.0",
  "id": "3f9a1c0b7d2e-0004",
  "source": {
    "document": "handbook.pdf",
    "chunk_index": 4,
    "text": "Employees accrue 1.5 vacation days per month. Unused days carry over for 12 months..."
  },
  "title": "Vacation accrual",
  "summary": "Employees earn vacation days monthly. Unused days expire after a year.",
  "facts": [
    "Employees accrue 1.5 vacation days per month.",
    "Unused vacation days carry over for 12 months."
  ],
  "tags": ["vacation", "leave", "accrual"],
  "questions": [
    "How many vacation days do employees earn?",
    "Do unused vacation days carry over?"
  ]
}
```

If the model returns malformed output for a chunk, a fallback record is built
(title = first line, summary = first sentences, facts = the sentences), so no
content is lost.

## How it works

1. **Indexing** (cached by file SHA-256 in `data/store.db`, tables `okf_documents` / `okf_records`):
   the document is split into larger chunks (`CHUNK_SIZE` x 1.5), `granite3.3:2b` turns each
   chunk into an OKF record, and the combined text (title + summary + facts + questions) is
   embedded with `nomic-embed-text:v1.5`.
2. **Query**: the question is embedded and every record is scored by cosine similarity plus a
   keyword bonus (token overlap with tags/title and facts). The top-k records form the
   context: title, summary and facts, plus the original source text for the top 2 records when
   it fits the context budget. The model answers only from these records, streaming tokens.

## Setup

```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
ollama pull granite3.3:2b
ollama pull nomic-embed-text:v1.5
```

All settings (models, chunk size, top-k, ...) are in `.env`; see `.env.example`.
No extra dependencies are needed: SQLite and numpy only.

## Usage

```cmd
python chatbot_okf.py path\to\document.pdf
python chatbot_okf.py            REM normal chatbot, load a document later with /load
```

| Command | Action |
|---|---|
| `/doc` | document mode |
| `/normal` | normal chat mode |
| `/load <path>` | load another document (builds OKF records on first load) |
| `/records` | show record count and titles |
| `/export` | write all records to `data/<docname>.okf.json` (portable OKF file) |
| `/clear` | clear history |
| `/help`, `/exit` | help / quit |

The exported file wraps the records:
`{"okf_version", "document", "sha256", "record_count", "records": [...]}`.

## Layout

| Path | Purpose |
|---|---|
| `chatbot_okf.py` | entry point: record store, OKF builder, retrieval, chat loop |
| `ollama_client.py` | Ollama HTTP client (embeddings, chat, JSON chat) |
| `document_loader.py` | file parsing and chunking |
| `config.py` | `.env` settings |
| `logger.py` | central logger (`logs/YYYY-MM-DD/<timestamp>.log`) |
| `exception.py` | custom exception hierarchy |
| `data/` | SQLite database and exported `.okf.json` files |
| `logs/` | run logs |
