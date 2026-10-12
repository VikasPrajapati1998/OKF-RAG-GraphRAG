"""Command-line RAG chatbot (Ollama + SQLite vector store).

Usage:
    python chatbot_rag.py [path/to/document]

Without a document the bot is a normal chatbot. With a document it chunks and
embeds the file (cached by content hash), then answers questions from it.
Type /help inside the chat for commands.
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np

from config import Settings, get_settings
from document_loader import chunk_text, file_sha256, load_document
from exception import ChatbotError, DatabaseError, RetrievalError
from logger import logger
from ollama_client import OllamaClient

_NORMAL_SYSTEM = "You are a helpful, concise assistant."
_DOC_SYSTEM = (
    "You answer questions strictly from the provided document context. "
    "If the answer is not in the context, say \"I couldn't find that in the document.\" "
    "Be concise and accurate; do not invent facts."
)
_HELP = (
    "Commands:\n"
    "  /doc            switch to document mode\n"
    "  /normal         switch to normal chat mode\n"
    "  /load <path>    load another document\n"
    "  /clear          clear conversation history\n"
    "  /help           show this help\n"
    "  /exit           quit"
)


class VectorStore:
    """SQLite-backed vector store with in-memory cosine search."""

    def __init__(self, db_path: Path) -> None:
        """Open (and initialise) the database.

        Args:
            db_path: SQLite file path.

        Raises:
            DatabaseError: If the database cannot be opened or initialised.
        """
        try:
            self._conn = sqlite3.connect(db_path)
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                    idx INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
                """
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Cannot open database {db_path}: {exc}") from exc

    def has_document(self, doc_id: str) -> bool:
        """Return True if the document is already indexed."""
        try:
            row = self._conn.execute("SELECT 1 FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Lookup failed: {exc}") from exc
        return row is not None

    def add_document(self, doc_id: str, name: str, chunks: list[str], vectors: np.ndarray) -> None:
        """Store a document with its chunk embeddings atomically.

        Args:
            doc_id: Content hash identifying the document.
            name: Display name.
            chunks: Chunk texts.
            vectors: Embedding matrix aligned with ``chunks``.

        Raises:
            DatabaseError: If the write fails.
        """
        try:
            with self._conn:
                self._conn.execute("INSERT OR REPLACE INTO documents (doc_id, name) VALUES (?, ?)", (doc_id, name))
                self._conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
                self._conn.executemany(
                    "INSERT INTO chunks (doc_id, idx, text, embedding) VALUES (?, ?, ?, ?)",
                    [
                        (doc_id, i, text, vectors[i].astype(np.float32).tobytes())
                        for i, text in enumerate(chunks)
                    ],
                )
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to store document {name}: {exc}") from exc

    def load_chunks(self, doc_id: str) -> tuple[list[str], np.ndarray]:
        """Load all chunks and embeddings of a document.

        Args:
            doc_id: Document identifier.

        Returns:
            Tuple of chunk texts and the embedding matrix.

        Raises:
            DatabaseError: If the read fails.
        """
        try:
            rows = self._conn.execute(
                "SELECT text, embedding FROM chunks WHERE doc_id = ? ORDER BY idx", (doc_id,)
            ).fetchall()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to load chunks: {exc}") from exc
        texts = [r[0] for r in rows]
        matrix = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows]) if rows else np.empty((0, 0))
        return texts, matrix

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()


class RagEngine:
    """Indexes a document and retrieves the most relevant chunks."""

    def __init__(self, settings: Settings, client: OllamaClient, store: VectorStore) -> None:
        """Create the engine.

        Args:
            settings: Application settings.
            client: Ollama client.
            store: Vector store.
        """
        self._settings = settings
        self._client = client
        self._store = store
        self._texts: list[str] = []
        self._matrix: np.ndarray = np.empty((0, 0))
        self.doc_name: str | None = None

    @property
    def ready(self) -> bool:
        """Whether a document is loaded."""
        return bool(self._texts)

    def load(self, path: Path) -> bool:
        """Index (or reuse the cached index of) a document.

        Args:
            path: Document path.

        Returns:
            True if embeddings were freshly computed, False if cached.

        Raises:
            ChatbotError: On any loading, embedding or database failure.
        """
        doc_id = file_sha256(path)
        fresh = not self._store.has_document(doc_id)
        if fresh:
            text = load_document(path)
            chunks = chunk_text(text, self._settings.chunk_size, self._settings.chunk_overlap)
            logger.info("Embedding %d chunks from %s", len(chunks), path.name)
            vectors = self._client.embed(chunks)
            self._store.add_document(doc_id, path.name, chunks, vectors)
        self._texts, self._matrix = self._store.load_chunks(doc_id)
        self.doc_name = path.name
        logger.info("Document ready: %s (%d chunks, fresh=%s)", path.name, len(self._texts), fresh)
        return fresh

    def retrieve(self, query: str) -> list[str]:
        """Return the top-k chunks most similar to the query.

        Args:
            query: Search text.

        Returns:
            Chunk texts ordered by relevance.

        Raises:
            RetrievalError: If no document is loaded.
        """
        if not self.ready:
            raise RetrievalError("No document loaded")
        query_vec = self._client.embed([query], is_query=True)[0]
        scores = self._matrix @ query_vec
        k = min(self._settings.top_k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        logger.debug("Retrieved chunks %s scores %s", top.tolist(), scores[top].round(3).tolist())
        return [self._texts[i] for i in top]


class Chatbot:
    """Interactive chat session supporting normal and document modes."""

    def __init__(self, settings: Settings, client: OllamaClient, engine: RagEngine) -> None:
        """Create a session.

        Args:
            settings: Application settings.
            client: Ollama client.
            engine: RAG engine.
        """
        self._settings = settings
        self._client = client
        self._engine = engine
        self._history: list[dict[str, str]] = []
        self.doc_mode = False

    def _trimmed_history(self) -> list[dict[str, str]]:
        """Return the most recent turns (two messages per turn)."""
        limit = self._settings.max_history_turns * 2
        return self._history[-limit:] if limit else []

    def _build_messages(self, question: str) -> list[dict[str, str]]:
        """Compose the prompt for the current mode."""
        if not self.doc_mode:
            return [{"role": "system", "content": _NORMAL_SYSTEM}, *self._trimmed_history(),
                    {"role": "user", "content": question}]
        search = question
        if len(question.split()) < 6 and self._history:  # short follow-up: add prior question for context
            search = f"{self._history[-2]['content']} {question}"
        context = "\n\n---\n\n".join(self._engine.retrieve(search))
        prompt = f"Document context:\n{context}\n\nQuestion: {question}\nAnswer:"
        return [{"role": "system", "content": _DOC_SYSTEM}, *self._trimmed_history(),
                {"role": "user", "content": prompt}]

    def ask(self, question: str) -> None:
        """Answer a question, streaming the reply to stdout.

        Args:
            question: The user's message.

        Raises:
            ChatbotError: If retrieval or generation fails.
        """
        messages = self._build_messages(question)
        parts: list[str] = []
        print("Bot: ", end="", flush=True)
        try:
            for token in self._client.chat_stream(messages):
                parts.append(token)
                print(token, end="", flush=True)
        finally:
            print()
        self._history += [{"role": "user", "content": question}, {"role": "assistant", "content": "".join(parts)}]

    def load_document(self, raw_path: str) -> None:
        """Load a document and switch to document mode.

        Args:
            raw_path: Path typed by the user.
        """
        path = Path(raw_path.strip().strip('"')).expanduser()
        print(f"Indexing {path.name} ...", flush=True)
        fresh = self._engine.load(path)
        self.doc_mode = True
        self._history.clear()
        print(f"Document ready ({'indexed' if fresh else 'loaded from cache'}). Document mode ON.")

    def handle_command(self, line: str) -> bool:
        """Process a slash command.

        Args:
            line: The raw input line starting with ``/``.

        Returns:
            False if the session should end, otherwise True.
        """
        command, _, argument = line.partition(" ")
        command = command.lower()
        if command in ("/exit", "/quit"):
            return False
        if command == "/help":
            print(_HELP)
        elif command == "/clear":
            self._history.clear()
            print("History cleared.")
        elif command == "/normal":
            self.doc_mode = False
            print("Normal chat mode ON.")
        elif command == "/doc":
            if self._engine.ready:
                self.doc_mode = True
                print(f"Document mode ON ({self._engine.doc_name}).")
            else:
                print("No document loaded. Use /load <path>.")
        elif command == "/load":
            if argument.strip():
                self.load_document(argument)
            else:
                print("Usage: /load <path>")
        else:
            print("Unknown command. Type /help.")
        return True

    def run(self) -> None:
        """Run the interactive loop until the user exits."""
        mode = f"document mode ({self._engine.doc_name})" if self.doc_mode else "normal mode"
        print(f"RAG chatbot ready in {mode}. Type /help for commands.")
        while True:
            try:
                line = input(f"\nYou [{'doc' if self.doc_mode else 'chat'}]: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            try:
                if line.startswith("/"):
                    if not self.handle_command(line):
                        break
                else:
                    self.ask(line)
            except ChatbotError as exc:
                logger.exception("Request failed")
                print(f"[error] {exc}")
            except KeyboardInterrupt:
                print("\n[interrupted]")


def main() -> int:
    """Program entry point.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="RAG chatbot over a local document (Ollama).")
    parser.add_argument("document", nargs="?", help="path to a .txt, .md, .csv, .pdf or .docx file")
    args = parser.parse_args()

    store: VectorStore | None = None
    try:
        settings = get_settings()
        client = OllamaClient(settings)
        client.check_ready()
        store = VectorStore(settings.db_path)
        bot = Chatbot(settings, client, RagEngine(settings, client, store))
        if args.document:
            bot.load_document(args.document)
        bot.run()
        return 0
    except ChatbotError as exc:
        logger.exception("Fatal error")
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    finally:
        if store is not None:
            store.close()


if __name__ == "__main__":
    sys.exit(main())
