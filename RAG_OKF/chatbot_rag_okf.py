"""Command-line hybrid RAG + OKF chatbot (Ollama + SQLite).

Usage:
    python chatbot_rag_okf.py [path/to/document]

Without a document the bot is a normal chatbot. With a document, the text is
chunked once and every chunk is indexed twice, linked 1:1: as a raw chunk
(RAG side) and as a structured OKF knowledge record built by the LLM (title,
summary, facts, tags, questions). Questions are answered by fusing both
rankings with Reciprocal Rank Fusion. Everything is cached by content hash.
Type /help inside the chat for commands.
"""

import argparse
import json
import re
import sqlite3
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

from config import Settings, get_settings
from document_loader import chunk_text, file_sha256, load_document
from exception import ChatbotError, DatabaseError, DocumentError, LLMError, RetrievalError
from logger import logger
from ollama_client import OllamaClient

OKF_VERSION = "1.0"

_RRF_K = 60  # Reciprocal Rank Fusion constant
_MAX_FALLBACK_FACTS = 20
_TITLE_WEIGHT = 0.25  # keyword bonus for query tokens found in title/tags
_FACT_WEIGHT = 0.15  # keyword bonus for query tokens found in facts
_MIN_TOKEN_LEN = 3
_RAW_TEXT_RESULTS = 2  # how many top fused chunks also contribute their raw text
_CONTEXT_CHARS_PER_CTX_TOKEN = 2  # context budget in characters = num_ctx * 2
_STOPWORDS = frozenset(
    "the and for are was were with that this from have has had not but you your what when where which who whom "
    "how why can could does did its their they them then than into about over under also any all".split()
)

_NORMAL_SYSTEM = "You are a helpful, concise assistant."
_DOC_SYSTEM = (
    "You answer questions strictly from the provided context (knowledge records and excerpts). "
    "Answer only from that context. "
    "If the answer is not in the context, say \"I couldn't find that in the document.\" "
    "Be concise and accurate; do not invent facts."
)
_EXTRACT_SYSTEM = (
    "You convert text into one JSON knowledge record. Use only information from the text. "
    "Reply with JSON only, in exactly this shape:\n"
    '{"title": "short title", "summary": "one or two sentences", '
    '"facts": ["atomic self-contained statement", "another statement"], '
    '"tags": ["keyword", "keyword"], '
    '"questions": ["question the text answers?", "another question?"]}\n'
    "Facts must be short, standalone sentences that name their subject (no pronouns). "
    "Give 2 or 3 questions."
)
_HELP = (
    "Commands:\n"
    "  /doc            switch to document mode\n"
    "  /normal         switch to normal chat mode\n"
    "  /load <path>    load another document\n"
    "  /records        show OKF record count and titles\n"
    "  /export         write the OKF records to data/<docname>.okf.json\n"
    "  /clear          clear conversation history\n"
    "  /help           show this help\n"
    "  /exit           quit"
)


def _as_text(value: object) -> str:
    """Coerce a JSON value to a stripped string (non-scalars become empty)."""
    return " ".join(value.split()) if isinstance(value, str) else (str(value) if isinstance(value, (int, float)) else "")


def _as_list(value: object) -> list[str]:
    """Coerce a JSON value to a list of unique, non-empty strings."""
    items = value if isinstance(value, list) else [value]
    seen: dict[str, None] = {}
    for item in items:
        text = _as_text(item)
        if text:
            seen.setdefault(text, None)
    return list(seen)


def _sentences(text: str) -> list[str]:
    """Split text into non-empty sentences/lines."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def _tokens(text: str) -> set[str]:
    """Lower-case alphanumeric tokens without stopwords and very short words."""
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) >= _MIN_TOKEN_LEN and t not in _STOPWORDS}


def _retrieval_text(record: dict) -> str:
    """Combine title, summary, facts and questions into the text that gets embedded."""
    return "\n".join([record["title"], record["summary"], *record["facts"], *record["questions"]])


def _rank_order(scores: np.ndarray) -> np.ndarray:
    """Return chunk indices sorted from best to worst score."""
    return np.argsort(-scores, kind="stable")


class HybridStore:
    """SQLite-backed store holding raw chunks and their linked OKF records."""

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
                CREATE TABLE IF NOT EXISTS hybrid_documents (
                    doc_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS hybrid_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL REFERENCES hybrid_documents(doc_id) ON DELETE CASCADE,
                    idx INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    record TEXT NOT NULL,
                    record_embedding BLOB NOT NULL,
                    fallback INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_hybrid_chunks_doc ON hybrid_chunks(doc_id);
                """
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Cannot open database {db_path}: {exc}") from exc

    def has_document(self, doc_id: str) -> bool:
        """Return True if the document is already indexed."""
        try:
            row = self._conn.execute("SELECT 1 FROM hybrid_documents WHERE doc_id = ?", (doc_id,)).fetchone()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Lookup failed: {exc}") from exc
        return row is not None

    def add_document(
        self,
        doc_id: str,
        name: str,
        chunks: list[str],
        vectors: np.ndarray,
        records: list[dict],
        record_vectors: np.ndarray,
        fallbacks: list[bool],
    ) -> None:
        """Store a document with chunks, records and both embedding sets atomically.

        Args:
            doc_id: Content hash identifying the document.
            name: Display name.
            chunks: Raw chunk texts.
            vectors: Chunk embedding matrix aligned with ``chunks``.
            records: OKF record documents aligned with ``chunks``.
            record_vectors: Record embedding matrix aligned with ``chunks``.
            fallbacks: Whether each record came from the non-LLM fallback.

        Raises:
            DatabaseError: If the write fails.
        """
        try:
            with self._conn:
                self._conn.execute("INSERT OR REPLACE INTO hybrid_documents (doc_id, name) VALUES (?, ?)", (doc_id, name))
                self._conn.execute("DELETE FROM hybrid_chunks WHERE doc_id = ?", (doc_id,))
                self._conn.executemany(
                    "INSERT INTO hybrid_chunks (doc_id, idx, text, embedding, record, record_embedding, fallback) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            doc_id,
                            i,
                            text,
                            vectors[i].astype(np.float32).tobytes(),
                            json.dumps(records[i], ensure_ascii=False),
                            record_vectors[i].astype(np.float32).tobytes(),
                            int(fallbacks[i]),
                        )
                        for i, text in enumerate(chunks)
                    ],
                )
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to store document {name}: {exc}") from exc

    def load_document(self, doc_id: str) -> tuple[list[str], np.ndarray, list[dict], np.ndarray, list[bool]]:
        """Load all chunks, records and embeddings of a document.

        Args:
            doc_id: Document identifier.

        Returns:
            Tuple of chunk texts, chunk embedding matrix, OKF records, record
            embedding matrix and per-record fallback flags (all in chunk order).

        Raises:
            DatabaseError: If the read fails or a stored record is corrupt.
        """
        try:
            rows = self._conn.execute(
                "SELECT text, embedding, record, record_embedding, fallback FROM hybrid_chunks "
                "WHERE doc_id = ? ORDER BY idx",
                (doc_id,),
            ).fetchall()
            records = [json.loads(r[2]) for r in rows]
        except (sqlite3.Error, ValueError) as exc:
            raise DatabaseError(f"Failed to load document: {exc}") from exc
        texts = [r[0] for r in rows]
        empty = np.empty((0, 0))
        matrix = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows]) if rows else empty
        record_matrix = np.vstack([np.frombuffer(r[3], dtype=np.float32) for r in rows]) if rows else empty
        return texts, matrix, records, record_matrix, [bool(r[4]) for r in rows]

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()


class OkfBuilder:
    """Turns text chunks into validated OKF records using the LLM."""

    def __init__(self, client: OllamaClient) -> None:
        """Create the builder.

        Args:
            client: Ollama client.
        """
        self._client = client

    @staticmethod
    def normalise(raw: dict) -> dict | None:
        """Validate and normalise raw model output.

        Args:
            raw: Parsed JSON object from the model.

        Returns:
            Dict with ``title``, ``summary``, ``facts``, ``tags`` and ``questions``,
            or None if the output has no usable content.
        """
        record = {
            "title": _as_text(raw.get("title")),
            "summary": _as_text(raw.get("summary")),
            "facts": _as_list(raw.get("facts")),
            "tags": [t.lower() for t in _as_list(raw.get("tags"))],
            "questions": _as_list(raw.get("questions")),
        }
        if not record["facts"] and not record["summary"]:
            return None
        if not record["title"]:
            record["title"] = record["summary"][:80] or record["facts"][0][:80]
        return record

    @staticmethod
    def fallback(chunk: str) -> dict:
        """Build a plain record from the chunk text so nothing is lost.

        Args:
            chunk: Source chunk text.

        Returns:
            A record whose title is the first line and facts are the sentences.
        """
        sentences = _sentences(chunk)
        first_line = chunk.strip().splitlines()[0].strip() if chunk.strip() else ""
        return {
            "title": first_line[:80],
            "summary": " ".join(sentences[:2]),
            "facts": sentences[:_MAX_FALLBACK_FACTS],
            "tags": [],
            "questions": [],
        }

    def build(self, chunk: str) -> tuple[dict, bool]:
        """Create an OKF record body for one chunk.

        Args:
            chunk: Source chunk text.

        Returns:
            Tuple of the normalised record and a flag that is True when the
            fallback record was used because the model output was unusable.
        """
        messages = [
            {"role": "system", "content": _EXTRACT_SYSTEM},
            {"role": "user", "content": f"Text:\n{chunk}\n\nJSON:"},
        ]
        try:
            record = self.normalise(self._client.chat_json(messages))
        except LLMError as exc:
            logger.warning("Record extraction failed (%s); using fallback record", exc)
            return self.fallback(chunk), True
        if record is None:
            logger.warning("Model returned an empty record; using fallback record")
            return self.fallback(chunk), True
        return record, False


class HybridEngine:
    """Indexes a document as chunks plus OKF records and retrieves with rank fusion."""

    def __init__(self, settings: Settings, client: OllamaClient, store: HybridStore) -> None:
        """Create the engine.

        Args:
            settings: Application settings.
            client: Ollama client.
            store: Hybrid store.
        """
        self._settings = settings
        self._client = client
        self._store = store
        self._builder = OkfBuilder(client)
        self._texts: list[str] = []
        self._matrix: np.ndarray = np.empty((0, 0))
        self._records: list[dict] = []
        self._record_matrix: np.ndarray = np.empty((0, 0))
        self._fallbacks: list[bool] = []
        self._head_tokens: list[set[str]] = []
        self._fact_tokens: list[set[str]] = []
        self._doc_id: str | None = None
        self.doc_name: str | None = None

    @property
    def ready(self) -> bool:
        """Whether a document is loaded."""
        return bool(self._texts)

    def load(self, path: Path, progress: Callable[[int, int], None] | None = None) -> bool:
        """Index (or reuse the cached index of) a document.

        Args:
            path: Document path.
            progress: Optional callback receiving (done, total) chunks while building records.

        Returns:
            True if the index was freshly built, False if cached.

        Raises:
            ChatbotError: On any loading, embedding or database failure.
        """
        doc_id = file_sha256(path)
        fresh = not self._store.has_document(doc_id)
        if fresh:
            text = load_document(path)
            chunks = chunk_text(text, self._settings.chunk_size, self._settings.chunk_overlap)
            logger.info("Building OKF records for %d chunks from %s", len(chunks), path.name)
            records: list[dict] = []
            fallbacks: list[bool] = []
            for i, chunk in enumerate(chunks):
                body, used_fallback = self._builder.build(chunk)
                records.append(
                    {
                        "okf_version": OKF_VERSION,
                        "id": f"{doc_id[:12]}-{i:04d}",
                        "source": {"document": path.name, "chunk_index": i, "text": chunk},
                        **body,
                    }
                )
                fallbacks.append(used_fallback)
                if progress:
                    progress(i + 1, len(chunks))
            logger.info("Embedding %d chunks and %d records", len(chunks), len(records))
            vectors = self._client.embed(chunks)
            record_vectors = self._client.embed([_retrieval_text(r) for r in records])
            self._store.add_document(doc_id, path.name, chunks, vectors, records, record_vectors, fallbacks)
        (
            self._texts,
            self._matrix,
            self._records,
            self._record_matrix,
            self._fallbacks,
        ) = self._store.load_document(doc_id)
        self._head_tokens = [_tokens(" ".join([r["title"], *r["tags"]])) for r in self._records]
        self._fact_tokens = [_tokens(" ".join([r["summary"], *r["facts"]])) for r in self._records]
        self._doc_id = doc_id
        self.doc_name = path.name
        logger.info("Document ready: %s (%d chunks, fresh=%s)", path.name, len(self._texts), fresh)
        return fresh

    def titles(self) -> list[str]:
        """Return the titles of all loaded records in document order."""
        return [r["title"] for r in self._records]

    def export(self) -> Path:
        """Write all loaded records to the portable ``<docname>.okf.json`` file.

        Returns:
            Path of the written file.

        Raises:
            RetrievalError: If no document is loaded.
            DocumentError: If the file cannot be written.
        """
        if not self.ready or self.doc_name is None:
            raise RetrievalError("No document loaded")
        target = self._settings.data_dir / f"{Path(self.doc_name).stem}.okf.json"
        payload = {
            "okf_version": OKF_VERSION,
            "document": self.doc_name,
            "sha256": self._doc_id,
            "record_count": len(self._records),
            "records": self._records,
        }
        try:
            target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            raise DocumentError(f"Cannot write {target}: {exc}") from exc
        logger.info("Exported %d records to %s", len(self._records), target)
        return target

    def _record_scores(self, query: str, query_vec: np.ndarray) -> np.ndarray:
        """Score records by cosine similarity plus the keyword/tag overlap bonus."""
        scores = self._record_matrix @ query_vec
        query_tokens = _tokens(query)
        if query_tokens:
            bonus = np.array(
                [
                    _TITLE_WEIGHT * len(query_tokens & head) / len(query_tokens)
                    + _FACT_WEIGHT * len(query_tokens & facts) / len(query_tokens)
                    for head, facts in zip(self._head_tokens, self._fact_tokens)
                ]
            )
            scores = scores + bonus
        return scores

    def retrieve(self, query: str) -> list[int]:
        """Return the top-k chunk indices by fusing the RAG and OKF rankings.

        Args:
            query: Search text.

        Returns:
            Chunk indices ordered by fused relevance (best first).

        Raises:
            RetrievalError: If no document is loaded.
        """
        if not self.ready:
            raise RetrievalError("No document loaded")
        query_vec = self._client.embed([query], is_query=True)[0]
        rag_order = _rank_order(self._matrix @ query_vec)
        okf_order = _rank_order(self._record_scores(query, query_vec))

        fused: dict[int, float] = {}
        for order in (rag_order, okf_order):
            for rank, index in enumerate(order.tolist()):
                fused[index] = fused.get(index, 0.0) + 1.0 / (_RRF_K + rank + 1)
        k = min(self._settings.top_k, len(fused))
        top = sorted(fused, key=lambda i: (-fused[i], i))[:k]
        logger.debug("Fused chunks %s scores %s", top, [round(fused[i], 4) for i in top])
        return top

    def format_context(self, indices: list[int]) -> str:
        """Render the selected chunks as prompt context within the character budget.

        Each chunk contributes its record (title, summary, facts). Raw chunk
        text is added for the top fused results only, or whenever the record
        came from the fallback (whose facts are just the raw sentences).
        Repeated facts and repeated raw text are dropped.

        Args:
            indices: Chunk indices ordered by fused relevance.

        Returns:
            The context string.
        """
        budget = self._settings.num_ctx * _CONTEXT_CHARS_PER_CTX_TOKEN
        blocks: list[str] = []
        seen: set[str] = set()
        used = 0
        for rank, index in enumerate(indices):
            record = self._records[index]
            raw = self._texts[index]
            use_raw = rank < _RAW_TEXT_RESULTS or self._fallbacks[index]
            lines = [f"[{record['title']}]"]
            if not self._fallbacks[index]:  # fallback facts only repeat the raw sentences, so skip them
                if record["summary"] and record["summary"] not in seen:
                    seen.add(record["summary"])
                    lines.append(f"Summary: {record['summary']}")
                for fact in record["facts"]:
                    if fact not in seen:
                        seen.add(fact)
                        lines.append(f"- {fact}")
            block = "\n".join(lines)
            if use_raw and raw not in seen:
                block += f"\nExcerpt: {raw}"
                seen.add(raw)
            remaining = budget - used
            if remaining <= 0:
                break
            if len(block) > remaining:
                if blocks:  # keep whole blocks except when even the first one is too large
                    break
                block = block[:remaining]
            blocks.append(block)
            used += len(block)
        return "\n\n---\n\n".join(blocks)


class Chatbot:
    """Interactive chat session supporting normal and document modes."""

    def __init__(self, settings: Settings, client: OllamaClient, engine: HybridEngine) -> None:
        """Create a session.

        Args:
            settings: Application settings.
            client: Ollama client.
            engine: Hybrid engine.
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
        context = self._engine.format_context(self._engine.retrieve(search))
        prompt = f"Context (knowledge records and excerpts):\n{context}\n\nQuestion: {question}\nAnswer:"
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

        def show(done: int, total: int) -> None:
            """Print indexing progress on one line."""
            print(f"\r  building OKF records {done}/{total}", end="" if done < total else "\n", flush=True)

        fresh = self._engine.load(path, progress=show)
        self.doc_mode = True
        self._history.clear()
        print(f"Document ready ({'indexed' if fresh else 'loaded from cache'}). Document mode ON.")

    def _show_records(self) -> None:
        """Print the record count and titles."""
        if not self._engine.ready:
            print("No document loaded. Use /load <path>.")
            return
        titles = self._engine.titles()
        print(f"{len(titles)} OKF records ({self._engine.doc_name}):")
        for i, title in enumerate(titles, 1):
            print(f"  {i:>3}. {title}")

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
        elif command == "/records":
            self._show_records()
        elif command == "/export":
            if self._engine.ready:
                print(f"Exported OKF records to {self._engine.export()}")
            else:
                print("No document loaded. Use /load <path>.")
        else:
            print("Unknown command. Type /help.")
        return True

    def run(self) -> None:
        """Run the interactive loop until the user exits."""
        mode = f"document mode ({self._engine.doc_name})" if self.doc_mode else "normal mode"
        print(f"Hybrid RAG+OKF chatbot ready in {mode}. Type /help for commands.")
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
    parser = argparse.ArgumentParser(description="Hybrid RAG + OKF chatbot over a local document (Ollama).")
    parser.add_argument("document", nargs="?", help="path to a .txt, .md, .csv, .pdf or .docx file")
    args = parser.parse_args()

    store: HybridStore | None = None
    try:
        settings = get_settings()
        client = OllamaClient(settings)
        client.check_ready()
        store = HybridStore(settings.db_path)
        bot = Chatbot(settings, client, HybridEngine(settings, client, store))
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
