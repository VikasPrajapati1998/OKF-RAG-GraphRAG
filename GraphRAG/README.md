# GraphRAG Chatbot

Command-line chatbot that answers questions about a document using GraphRAG:
an LLM-built knowledge graph combined with vector retrieval. Runs fully locally
with Ollama, SQLite and numpy; no UI and no graph database.

## How it works

### Graph building (once per document)

1. The document (`.txt`, `.md`, `.csv`, `.pdf`, `.docx`) is split into overlapping chunks.
2. For every chunk `granite3.3:2b` extracts entities (`name`, `type`, `description`) and
   relations (`source`, `target`, `relation`) as strict JSON. Malformed output for a chunk is
   logged as a warning and skipped.
3. Entity names are normalised (lowercase, trimmed) so the same entity across chunks is merged;
   descriptions are merged and every node/edge remembers its source chunks.
4. Entities (`name: description`) and chunks are embedded with `nomic-embed-text:v1.5`.
5. Chunks, nodes, edges and embeddings are saved in SQLite (`data/store.db`), cached by file
   SHA-256, so re-opening the same document skips extraction entirely. Extraction is the slow
   step and runs sequentially (Ollama serialises requests anyway); a progress line is shown.

### Answering a question

1. The question is embedded.
2. Seed entities: the most similar entities (cosine) plus entities whose names appear in the question.
3. The graph is expanded one hop in memory to collect `A -[relation]-> B` facts and entity descriptions.
4. Chunks are retrieved by vector similarity (top-k) plus a few chunks linked to the seed entities.
5. A compact context is assembled within a character budget (facts first, then excerpts) and
   the model answers strictly from it, streaming tokens as they are generated.

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

## Usage

```cmd
python chatbot_graphrag.py path\to\document.pdf
python chatbot_graphrag.py            REM normal chatbot, load a document later with /load
```

| Command | Action |
|---|---|
| `/doc` | document mode |
| `/normal` | normal chat mode |
| `/load <path>` | load another document (builds its graph) |
| `/graph` | node/edge counts and top entities by degree |
| `/clear` | clear history |
| `/help`, `/exit` | help / quit |

## Layout

| Path | Purpose |
|---|---|
| `chatbot_graphrag.py` | entry point: graph store, graph extraction and retrieval, chat loop |
| `ollama_client.py` | Ollama HTTP client (embeddings, chat, JSON chat) |
| `document_loader.py` | file parsing and chunking |
| `config.py` | `.env` settings |
| `logger.py` | central logger (`logs/YYYY-MM-DD/<timestamp>.log`) |
| `exception.py` | custom exception hierarchy |
| `data/` | SQLite database (documents, chunks, entities, entity_chunks, edges) |
| `logs/` | run logs |
