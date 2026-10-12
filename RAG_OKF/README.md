# Hybrid RAG + OKF Chatbot

Command-line chatbot that answers questions about a document by combining classic
Retrieval-Augmented Generation (raw chunks) with the Open Knowledge Format (LLM-built
structured records). Runs fully locally with Ollama; no UI.

## How it works

1. The document (`.txt`, `.md`, `.csv`, `.pdf`, `.docx`) is split once into overlapping chunks.
2. Every chunk is indexed twice, linked 1:1 by chunk index, in SQLite (`data/store.db`):
   - **RAG side**: the raw chunk text and its embedding.
   - **OKF side**: a knowledge record (`okf_version` "1.0": title, summary, facts, tags,
     questions) built by the LLM, plus an embedding of its combined retrieval text. Malformed
     model output falls back to a plain record built from the chunk sentences.
   
   Everything is cached by file SHA-256, so re-opening the same document is instant.
3. Each question is embedded once and produces two rankings over the chunks:
   - **R1**: cosine similarity against the raw chunk embeddings.
   - **R2**: cosine similarity against the record embeddings plus a keyword/tag overlap bonus.
4. The rankings are merged with Reciprocal Rank Fusion (`score = sum(1 / (60 + rank))`) and the
   top `TOP_K` chunks are kept.
5. Context is built per chunk from the record (title, summary, facts); the raw chunk text is
   added for the top 2 fused results only, or when the record came from the fallback. Repeated
   content is dropped and the context stays within about `NUM_CTX * 2` characters.
6. `granite3.3:2b` answers strictly from that context, streaming tokens as they are generated.
   Short follow-up questions are searched together with the previous question.

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
python chatbot_rag_okf.py path\to\document.pdf
python chatbot_rag_okf.py            REM normal chatbot, load a document later with /load
```

The first load of a document calls the LLM once per chunk to build records (a one-line
progress counter is shown); later loads use the cache.

| Command | Action |
|---|---|
| `/doc` | document mode |
| `/normal` | normal chat mode |
| `/load <path>` | load another document |
| `/records` | show OKF record count and titles |
| `/export` | write the records to `data/<docname>.okf.json` |
| `/clear` | clear history |
| `/help`, `/exit` | help / quit |

## Layout

| Path | Purpose |
|---|---|
| `chatbot_rag_okf.py` | entry point: hybrid store, record builder, rank fusion, chat loop |
| `ollama_client.py` | Ollama HTTP client (embeddings, chat, JSON chat) |
| `document_loader.py` | file parsing and chunking |
| `config.py` | `.env` settings |
| `logger.py` | central logger (`logs/YYYY-MM-DD/<timestamp>.log`) |
| `exception.py` | custom exception hierarchy |
| `data/` | SQLite database and OKF exports |
| `logs/` | run logs |
