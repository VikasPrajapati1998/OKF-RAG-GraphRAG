# RAG vs GraphRAG vs OKF chatbots

Four CLI-only chatbots that run locally on Ollama and answer questions about your document. Each also works as a normal chatbot.

| Directory | Approach | Run |
|---|---|---|
| [`RAG/`](RAG/README.md) | chunk embeddings in SQLite, cosine top-k | `python chatbot_rag.py doc.pdf` |
| [`GraphRAG/`](GraphRAG/README.md) | LLM-extracted knowledge graph + vectors | `python chatbot_graphrag.py doc.pdf` |
| [`OKF/`](OKF/README.md) | Open Knowledge Format records (title, summary, facts, tags) | `python chatbot_okf.py doc.pdf` |
| [`RAG_OKF/`](RAG_OKF/README.md) | hybrid: raw-chunk vectors + OKF records, rank-fused | `python chatbot_rag_okf.py doc.pdf` |

Models: `granite3.3:2b` (chat) and `nomic-embed-text:v1.5` (embeddings). See `.claude/SETUP.md` for setup.
