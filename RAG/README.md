# RAG Chatbot

Command-line chatbot that answers questions about a document using classic
Retrieval-Augmented Generation. Runs fully locally with Ollama; no UI.

## How it works

1. The document (`.txt`, `.md`, `.csv`, `.pdf`, `.docx`) is split into overlapping chunks.
2. Chunks are embedded with `nomic-embed-text:v1.5` and stored in SQLite (`data/store.db`).
   Embeddings are cached by file hash, so re-opening the same document is instant.
3. Each question is embedded, the top-k similar chunks are retrieved (cosine similarity),
   and `granite3.3:2b` answers from them, streaming tokens as they are generated.

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
python chatbot_rag.py path\to\document.pdf
python chatbot_rag.py            REM normal chatbot, load a document later with /load
```

| Command | Action |
|---|---|
| `/doc` | document mode |
| `/normal` | normal chat mode |
| `/load <path>` | load another document |
| `/clear` | clear history |
| `/help`, `/exit` | help / quit |

## Layout

| Path | Purpose |
|---|---|
| `chatbot_rag.py` | entry point: vector store, retrieval, chat loop |
| `ollama_client.py` | Ollama HTTP client (embeddings, chat) |
| `document_loader.py` | file parsing and chunking |
| `config.py` | `.env` settings |
| `logger.py` | central logger (`logs/YYYY-MM-DD/<timestamp>.log`) |
| `exception.py` | custom exception hierarchy |
| `data/` | SQLite database |
| `logs/` | run logs |
