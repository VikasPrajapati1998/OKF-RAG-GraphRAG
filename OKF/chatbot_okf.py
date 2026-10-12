"""Command-line OKF chatbot (Open Knowledge Format, Ollama + SQLite).

Usage:
    python chatbot_okf.py [path/to/document]

Without a document the bot is a normal chatbot. With a document, every chunk is
converted by the LLM into a structured, self-describing OKF knowledge record
(title, summary, atomic facts, tags, questions). Records are cached by content
hash and retrieved instead of raw chunks. Type /help inside the chat for commands.
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

_CHUNK_SCALE = 1.5  # OKF chunks are larger: the LLM condenses them into records
_MAX_FALLBACK_FACTS = 20
_TITLE_WEIGHT = 0.25  # keyword bonus for query tokens found in title/tags
_FACT_WEIGHT = 0.15  # keyword bonus for query tokens found in facts
_MIN_TOKEN_LEN = 3
_SOURCE_RECORDS = 2  # how many top records may also contribute their source text
_CHARS_PER_TOKEN = 4
_STOPWORDS = frozenset(
    "the and for are was were with that this from have has had not but you your what when where which who whom "
    "how why can could does did its their they them then than into about over under also any all".split()
)

_NORMAL_SYSTEM = "You are a helpful, concise assistant."
_DOC_SYSTEM = (
    "You answer questions strictly from the provided knowledge records. "
    "If the answer is not in the records, say \"I couldn't find that in the document.\" "
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


class OkfStore:
    """SQLite-backed store for OKF records and their embeddings."""

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
                CREATE TABLE IF NOT EXISTS okf_documents (
                    doc_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS okf_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL REFERENCES okf_documents(doc_id) ON DELETE CASCADE,
                    idx INTEGER NOT NULL,
                    record TEXT NOT NULL,
                    embedding BLOB NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_okf_records_doc ON okf_records(doc_id);
                """
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Cannot open database {db_path}: {exc}") from exc

    def has_document(self, doc_id: str) -> bool:
        """Return True if the document is already indexed."""
        try:
            row = self._conn.execute("SELECT 1 FROM okf_documents WHERE doc_id = ?", (doc_id,)).fetchone()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Lookup failed: {exc}") from exc
        return row is not None

    def add_document(self, doc_id: str, name: str, records: list[dict], vectors: np.ndarray) -> None:
        """Store a document with its OKF records and embeddings atomically.

        Args:
            doc_id: Content hash identifying the document.
            name: Display name.
            records: OKF record documents.
            vectors: Embedding matrix aligned with ``records``.

        Raises:
            DatabaseError: If the write fails.
        """
        try:
            with self._conn:
                self._conn.execute("INSERT OR REPLACE INTO okf_documents (doc_id, name) VALUES (?, ?)", (doc_id, name))
                self._conn.execute("DELETE FROM okf_records WHERE doc_id = ?", (doc_id,))
                self._conn.executemany(
                    "INSERT INTO okf_records (doc_id, idx, record, embedding) VALUES (?, ?, ?, ?)",
                    [
                        (doc_id, i, json.dumps(rec, ensure_ascii=False), vectors[i].astype(np.float32).tobytes())
                        for i, rec in enumerate(records)
                    ],
                )
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to store records for {name}: {exc}") from exc

    def load_records(self, doc_id: str) -> tuple[list[dict], np.ndarray]:
        """Load all OKF records and embeddings of a document.

        Args:
            doc_id: Document identifier.

        Returns:
            Tuple of record documents and the embedding matrix.

        Raises:
            DatabaseError: If the read fails or a stored record is corrupt.
        """
        try:
            rows = self._conn.execute(
                "SELECT record, embedding FROM okf_records WHERE doc_id = ? ORDER BY idx", (doc_id,)
            ).fetchall()
            records = [json.loads(r[0]) for r in rows]
        except (sqlite3.Error, ValueError) as exc:
            raise DatabaseError(f"Failed to load records: {exc}") from exc
        matrix = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows]) if rows else np.empty((0, 0))
        return records, matrix

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

    def build(self, chunk: str) -> dict:
        """Create an OKF record body for one chunk.

        Args:
            chunk: Source chunk text.

        Returns:
            A normalised record; a fallback record if the model output is unusable.
        """
        messages = [
            {"role": "system", "content": _EXTRACT_SYSTEM},
            {"role": "user", "content": f"Text:\n{chunk}\n\nJSON:"},
        ]
        try:
            record = self.normalise(self._client.chat_json(messages))
        except LLMError as exc:
            logger.warning("Record extraction failed (%s); using fallback record", exc)
            return self.fallback(chunk)
        if record is None:
            logger.warning("Model returned an empty record; using fallback record")
            return self.fallback(chunk)
        return record


class OkfEngine:
    """Indexes a document into OKF records and retrieves the most relevant ones."""

    def __init__(self, settings: Settings, client: OllamaClient, store: OkfStore) -> None:
        """Create the engine.

        Args:
            settings: Application settings.
            client: Ollama client.
            store: Record store.
        """
        self._settings = settings
        self._client = client
        self._store = store
        self._builder = OkfBuilder(client)
        self._records: list[dict] = []
        self._matrix: np.ndarray = np.empty((0, 0))
        self._head_tokens: list[set[str]] = []
        self._fact_tokens: list[set[str]] = []
        self._doc_id: str | None = None
        self.doc_name: str | None = None

    @property
    def ready(self) -> bool:
        """Whether a document is loaded."""
        return bool(self._records)

    def load(self, path: Path, progress: Callable[[int, int], None] | None = None) -> bool:
        """Index (or reuse the cached records of) a document.

        Args:
            path: Document path.
            progress: Optional callback receiving (done, total) chunks while indexing.

        Returns:
            True if records were freshly built, False if cached.

        Raises:
            ChatbotError: On any loading, embedding or database failure.
        """
        doc_id = file_sha256(path)
        fresh = not self._store.has_document(doc_id)
        if fresh:
            text = load_document(path)
            size = int(self._settings.chunk_size * _CHUNK_SCALE)
            chunks = chunk_text(text, size, self._settings.chunk_overlap)
            logger.info("Building OKF records for %d chunks from %s", len(chunks), path.name)
            records: list[dict] = []
            for i, chunk in enumerate(chunks):
                body = self._builder.build(chunk)
                records.append(
                    {
                        "okf_version": OKF_VERSION,
                        "id": f"{doc_id[:12]}-{i:04d}",
                        "source": {"document": path.name, "chunk_index": i, "text": chunk},
                        **body,
                    }
                )
                if progress:
                    progress(i + 1, len(chunks))
            vectors = self._client.embed([_retrieval_text(r) for r in records])
            self._store.add_document(doc_id, path.name, records, vectors)
        self._records, self._matrix = self._store.load_records(doc_id)
        self._head_tokens = [_tokens(" ".join([r["title"], *r["tags"]])) for r in self._records]
        self._fact_tokens = [_tokens(" ".join([r["summary"], *r["facts"]])) for r in self._records]
        self._doc_id = doc_id
        self.doc_name = path.name
        logger.info("Document ready: %s (%d records, fresh=%s)", path.name, len(self._records), fresh)
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

    def retrieve(self, query: str) -> list[dict]:
        """Return the top-k records for the query (cosine + keyword overlap).

        Args:
            query: Search text.

        Returns:
            Records ordered by relevance.

        Raises:
            RetrievalError: If no document is loaded.
        """
        if not self.ready:
            raise RetrievalError("No document loaded")
        query_vec = self._client.embed([query], is_query=True)[0]
        scores = self._matrix @ query_vec
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
        k = min(self._settings.top_k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        logger.debug("Retrieved records %s scores %s", top.tolist(), scores[top].round(3).tolist())
        return [self._records[i] for i in top]


class Chatbot:
    """Interactive chat session supporting normal and document modes."""

    def __init__(self, settings: Settings, client: OllamaClient, engine: OkfEngine) -> None:
        """Create a session.

        Args:
            settings: Application settings.
            client: Ollama client.
            engine: OKF engine.
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

    def _format_context(self, records: list[dict]) -> str:
        """Render records as prompt context; add source text for the top records if budget allows."""
        budget = self._settings.num_ctx * _CHARS_PER_TOKEN // 2
        blocks: list[str] = []
        used = 0
        for rank, record in enumerate(records):
            lines = [f"[{record['title']}]", f"Summary: {record['summary']}"]
            lines += [f"- {fact}" for fact in record["facts"]]
            block = "\n".join(lines)
            source = record["source"]["text"]
            if rank < _SOURCE_RECORDS and used + len(block) + len(source) <= budget:
                block += f"\nSource text: {source}"
            blocks.append(block)
            used += len(block)
        return "\n\n---\n\n".join(blocks)

    def _build_messages(self, question: str) -> list[dict[str, str]]:
        """Compose the prompt for the current mode."""
        if not self.doc_mode:
            return [{"role": "system", "content": _NORMAL_SYSTEM}, *self._trimmed_history(),
                    {"role": "user", "content": question}]
        search = question
        if len(question.split()) < 6 and self._history:  # short follow-up: add prior question for context
            search = f"{self._history[-2]['content']} {question}"
        context = self._format_context(self._engine.retrieve(search))
        prompt = f"Knowledge records:\n{context}\n\nQuestion: {question}\nAnswer:"
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
        print(f"OKF chatbot ready in {mode}. Type /help for commands.")
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
    parser = argparse.ArgumentParser(description="OKF chatbot over a local document (Ollama).")
    parser.add_argument("document", nargs="?", help="path to a .txt, .md, .csv, .pdf or .docx file")
    args = parser.parse_args()

    store: OkfStore | None = None
    try:
        settings = get_settings()
        client = OllamaClient(settings)
        client.check_ready()
        store = OkfStore(settings.db_path)
        bot = Chatbot(settings, client, OkfEngine(settings, client, store))
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
