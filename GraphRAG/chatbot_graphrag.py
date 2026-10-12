"""Command-line GraphRAG chatbot (Ollama + SQLite knowledge graph).

Usage:
    python chatbot_graphrag.py [path/to/document]

Without a document the bot is a normal chatbot. With a document it builds a
knowledge graph (entities and relations extracted by the LLM, cached by content
hash) plus chunk embeddings, then answers questions from graph facts and the
most relevant chunks. Type /help inside the chat for commands.
"""

import argparse
import sqlite3
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from config import Settings, get_settings
from document_loader import chunk_text, file_sha256, load_document
from exception import ChatbotError, DatabaseError, LLMError, RetrievalError
from logger import logger
from ollama_client import OllamaClient

_NORMAL_SYSTEM = "You are a helpful, concise assistant."
_DOC_SYSTEM = (
    "You answer questions strictly from the provided knowledge-graph context "
    "(graph facts, entity notes and document excerpts). "
    "If the answer is not in the context, say \"I couldn't find that in the document.\" "
    "Be concise and accurate; do not invent facts."
)
_EXTRACT_SYSTEM = (
    "Extract a knowledge graph from the text. Reply with JSON only, exactly in this shape: "
    '{"entities":[{"name":"","type":"","description":""}],'
    '"relations":[{"source":"","target":"","relation":""}]} '
    "Use short names, at most 8 entities, descriptions under 15 words, "
    "relation as a short verb phrase; source and target must be entity names."
)
_HELP = (
    "Commands:\n"
    "  /doc            switch to document mode\n"
    "  /normal         switch to normal chat mode\n"
    "  /load <path>    load another document\n"
    "  /graph          show knowledge graph statistics\n"
    "  /clear          clear conversation history\n"
    "  /help           show this help\n"
    "  /exit           quit"
)

_MAX_ENTITIES_PER_CHUNK = 8
_MAX_DESCRIPTION_CHARS = 300
_MAX_NAME_CHARS = 80
_MAX_RELATION_CHARS = 60
_SEED_ENTITIES = 5  # entities picked by embedding similarity
_MIN_NAME_MATCH_CHARS = 3  # shortest entity name matched literally in the question
_MAX_FACTS_PER_SEED = 8
_EXTRA_LINKED_CHUNKS = 2  # chunks linked to matched entities, beyond the top-k
_CONTEXT_CHARS = 4000  # total character budget for the retrieved context
_FACTS_SHARE = 0.4  # share of the budget reserved for graph facts and entity notes
_TOP_ENTITIES_SHOWN = 10


def _normalise(name: str) -> str:
    """Return the merge key of an entity name (lowercase, collapsed whitespace)."""
    return " ".join(name.lower().split())


@dataclass
class _EntityDraft:
    """Entity being accumulated during extraction (merged across chunks)."""

    name: str
    type: str = ""
    descriptions: list[str] = field(default_factory=list)
    chunks: set[int] = field(default_factory=set)

    def add_description(self, text: str) -> None:
        """Append a description unless it is empty or already covered."""
        text = text.strip()
        if text and not any(text.lower() in d.lower() for d in self.descriptions):
            self.descriptions.append(text)

    def merged_description(self) -> str:
        """Return all descriptions joined and capped in length."""
        return "; ".join(self.descriptions)[:_MAX_DESCRIPTION_CHARS]


@dataclass
class _GraphDraft:
    """Entities and relations collected from every chunk of one document."""

    entities: dict[str, _EntityDraft] = field(default_factory=dict)
    edges: dict[tuple[str, str, str], set[int]] = field(default_factory=dict)

    def entity(self, name: str) -> _EntityDraft:
        """Get or create the entity for a (display) name."""
        key = _normalise(name)
        if key not in self.entities:
            self.entities[key] = _EntityDraft(name=name.strip())
        return self.entities[key]


class GraphStore:
    """SQLite persistence for chunks and the knowledge graph."""

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
                CREATE TABLE IF NOT EXISTS entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                    key TEXT NOT NULL,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    embedding BLOB NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_entities_doc ON entities(doc_id);
                CREATE TABLE IF NOT EXISTS entity_chunks (
                    entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                    chunk_idx INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_entity_chunks ON entity_chunks(entity_id);
                CREATE TABLE IF NOT EXISTS edges (
                    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                    src INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                    dst INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                    relation TEXT NOT NULL,
                    chunk_idx INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_edges_doc ON edges(doc_id);
                """
            )
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.commit()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Cannot open database {db_path}: {exc}") from exc

    def has_document(self, doc_id: str) -> bool:
        """Return True if the document (chunks and graph) is already indexed."""
        try:
            row = self._conn.execute("SELECT 1 FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Lookup failed: {exc}") from exc
        return row is not None

    def add_document(
        self,
        doc_id: str,
        name: str,
        chunks: list[str],
        chunk_vectors: np.ndarray,
        graph: _GraphDraft,
        entity_vectors: np.ndarray,
    ) -> None:
        """Store a document, its chunks and its knowledge graph atomically.

        Args:
            doc_id: Content hash identifying the document.
            name: Display name.
            chunks: Chunk texts.
            chunk_vectors: Embedding matrix aligned with ``chunks``.
            graph: Merged entities and relations.
            entity_vectors: Embedding matrix aligned with ``graph.entities`` order.

        Raises:
            DatabaseError: If the write fails.
        """
        try:
            with self._conn:
                self._conn.execute("INSERT INTO documents (doc_id, name) VALUES (?, ?)", (doc_id, name))
                self._conn.executemany(
                    "INSERT INTO chunks (doc_id, idx, text, embedding) VALUES (?, ?, ?, ?)",
                    [
                        (doc_id, i, text, chunk_vectors[i].astype(np.float32).tobytes())
                        for i, text in enumerate(chunks)
                    ],
                )
                ids: dict[str, int] = {}
                for i, (key, ent) in enumerate(graph.entities.items()):
                    cursor = self._conn.execute(
                        "INSERT INTO entities (doc_id, key, name, type, description, embedding) VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            doc_id,
                            key,
                            ent.name,
                            ent.type,
                            ent.merged_description(),
                            entity_vectors[i].astype(np.float32).tobytes(),
                        ),
                    )
                    ids[key] = int(cursor.lastrowid)
                    self._conn.executemany(
                        "INSERT INTO entity_chunks (entity_id, chunk_idx) VALUES (?, ?)",
                        [(ids[key], c) for c in sorted(ent.chunks)],
                    )
                self._conn.executemany(
                    "INSERT INTO edges (doc_id, src, dst, relation, chunk_idx) VALUES (?, ?, ?, ?, ?)",
                    [
                        (doc_id, ids[src], ids[dst], relation, c)
                        for (src, dst, relation), chunk_set in graph.edges.items()
                        for c in sorted(chunk_set)
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

    def load_graph(self, doc_id: str) -> "KnowledgeGraph":
        """Load the knowledge graph of a document into memory.

        Args:
            doc_id: Document identifier.

        Returns:
            The in-memory graph with adjacency lists.

        Raises:
            DatabaseError: If the read fails.
        """
        try:
            ent_rows = self._conn.execute(
                "SELECT id, name, description, embedding FROM entities WHERE doc_id = ? ORDER BY id", (doc_id,)
            ).fetchall()
            link_rows = self._conn.execute(
                "SELECT ec.entity_id, ec.chunk_idx FROM entity_chunks ec "
                "JOIN entities e ON e.id = ec.entity_id WHERE e.doc_id = ?",
                (doc_id,),
            ).fetchall()
            edge_rows = self._conn.execute(
                "SELECT src, dst, relation, chunk_idx FROM edges WHERE doc_id = ?", (doc_id,)
            ).fetchall()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to load graph: {exc}") from exc

        position = {row[0]: i for i, row in enumerate(ent_rows)}
        graph = KnowledgeGraph(
            names=[r[1] for r in ent_rows],
            descriptions=[r[2] for r in ent_rows],
            matrix=(
                np.vstack([np.frombuffer(r[3], dtype=np.float32) for r in ent_rows])
                if ent_rows
                else np.empty((0, 0))
            ),
        )
        for entity_id, chunk_idx in link_rows:
            graph.entity_chunks[position[entity_id]].add(chunk_idx)
        seen: set[tuple[int, int, str]] = set()
        for src, dst, relation, _chunk_idx in edge_rows:
            edge = (position[src], position[dst], relation)
            if edge in seen:  # one row per source chunk is stored; the graph needs each fact once
                continue
            seen.add(edge)
            graph.add_edge(*edge)
        return graph

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()


class KnowledgeGraph:
    """In-memory view of a document's knowledge graph."""

    def __init__(self, names: list[str], descriptions: list[str], matrix: np.ndarray) -> None:
        """Create a graph.

        Args:
            names: Entity display names.
            descriptions: Merged entity descriptions, aligned with ``names``.
            matrix: Entity embedding matrix, aligned with ``names``.
        """
        self.names = names
        self.descriptions = descriptions
        self.matrix = matrix
        self.entity_chunks: defaultdict[int, set[int]] = defaultdict(set)
        self.adjacency: defaultdict[int, list[tuple[int, str, bool]]] = defaultdict(list)
        self.edge_count = 0

    def add_edge(self, src: int, dst: int, relation: str) -> None:
        """Register a directed edge in both adjacency lists."""
        self.adjacency[src].append((dst, relation, True))
        self.adjacency[dst].append((src, relation, False))
        self.edge_count += 1

    def degree(self, node: int) -> int:
        """Return the number of edges touching a node."""
        return len(self.adjacency.get(node, ()))

    def facts(self, node: int, limit: int) -> list[str]:
        """Return up to ``limit`` ``A -[rel]-> B`` lines touching a node."""
        lines: list[str] = []
        for other, relation, outgoing in self.adjacency.get(node, ())[:limit]:
            a, b = (node, other) if outgoing else (other, node)
            lines.append(f"{self.names[a]} -[{relation}]-> {self.names[b]}")
        return lines

    def summary(self) -> str:
        """Return node/edge counts and the highest-degree entities."""
        top = sorted(range(len(self.names)), key=self.degree, reverse=True)[:_TOP_ENTITIES_SHOWN]
        lines = [f"Knowledge graph: {len(self.names)} entities, {self.edge_count} relations."]
        if top:
            lines.append("Top entities by degree:")
            lines += [f"  {self.degree(i):>3}  {self.names[i]}" for i in top]
        return "\n".join(lines)


class GraphRagEngine:
    """Builds the knowledge graph of a document and retrieves graph-aware context."""

    def __init__(self, settings: Settings, client: OllamaClient, store: GraphStore) -> None:
        """Create the engine.

        Args:
            settings: Application settings.
            client: Ollama client.
            store: Graph and vector store.
        """
        self._settings = settings
        self._client = client
        self._store = store
        self._texts: list[str] = []
        self._matrix: np.ndarray = np.empty((0, 0))
        self._graph = KnowledgeGraph([], [], np.empty((0, 0)))
        self.doc_name: str | None = None

    @property
    def ready(self) -> bool:
        """Whether a document is loaded."""
        return bool(self._texts)

    def graph_summary(self) -> str:
        """Return a printable summary of the loaded graph.

        Raises:
            RetrievalError: If no document is loaded.
        """
        if not self.ready:
            raise RetrievalError("No document loaded")
        return self._graph.summary()

    def _extract_chunk(self, chunk: str, index: int, draft: _GraphDraft) -> None:
        """Extract entities/relations of one chunk and merge them into ``draft``.

        Args:
            chunk: Chunk text.
            index: Chunk index, recorded as the source of every fact.
            draft: Accumulating graph.

        Raises:
            LLMError: If the model request fails or returns invalid JSON.
        """
        data = self._client.chat_json(
            [{"role": "system", "content": _EXTRACT_SYSTEM}, {"role": "user", "content": chunk}]
        )
        entities = data.get("entities")
        relations = data.get("relations")
        for item in entities[:_MAX_ENTITIES_PER_CHUNK] if isinstance(entities, list) else []:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                continue
            name = item["name"].strip()[:_MAX_NAME_CHARS]
            if not name:
                continue
            entity = draft.entity(name)
            entity.chunks.add(index)
            if isinstance(item.get("type"), str) and not entity.type:
                entity.type = item["type"].strip()[:40]
            if isinstance(item.get("description"), str):
                entity.add_description(item["description"][:_MAX_DESCRIPTION_CHARS])
        for item in relations if isinstance(relations, list) else []:
            if not isinstance(item, dict):
                continue
            source, target, relation = item.get("source"), item.get("target"), item.get("relation")
            if not all(isinstance(v, str) and v.strip() for v in (source, target, relation)):
                continue
            src_key, dst_key = _normalise(source), _normalise(target)
            if src_key == dst_key:
                continue
            draft.entity(source[:_MAX_NAME_CHARS]).chunks.add(index)
            draft.entity(target[:_MAX_NAME_CHARS]).chunks.add(index)
            edge = (_normalise(source[:_MAX_NAME_CHARS]), _normalise(target[:_MAX_NAME_CHARS]),
                    " ".join(relation.lower().split())[:_MAX_RELATION_CHARS])
            draft.edges.setdefault(edge, set()).add(index)

    def _build_graph(self, chunks: list[str], progress: Callable[[int, int], None] | None) -> _GraphDraft:
        """Run extraction over all chunks, tolerating per-chunk failures.

        Raises:
            LLMError: If extraction failed for every chunk.
        """
        draft = _GraphDraft()
        failures = 0
        for i, chunk in enumerate(chunks):
            try:
                self._extract_chunk(chunk, i, draft)
            except LLMError as exc:
                failures += 1
                logger.warning("Extraction failed for chunk %d/%d: %s", i + 1, len(chunks), exc)
            if progress:
                progress(i + 1, len(chunks))
        if failures == len(chunks):
            raise LLMError("Entity extraction failed for every chunk; check the model and Ollama logs")
        logger.info(
            "Extracted %d entities, %d relations (%d/%d chunks failed)",
            len(draft.entities), len(draft.edges), failures, len(chunks),
        )
        return draft

    def load(self, path: Path, progress: Callable[[int, int], None] | None = None) -> bool:
        """Index (or reuse the cached index of) a document.

        Args:
            path: Document path.
            progress: Optional callback ``(done, total)`` called after each extracted chunk.

        Returns:
            True if the document was freshly indexed, False if cached.

        Raises:
            ChatbotError: On any loading, extraction, embedding or database failure.
        """
        doc_id = file_sha256(path)
        fresh = not self._store.has_document(doc_id)
        if fresh:
            text = load_document(path)
            chunks = chunk_text(text, self._settings.chunk_size, self._settings.chunk_overlap)
            logger.info("Extracting graph from %d chunks of %s", len(chunks), path.name)
            draft = self._build_graph(chunks, progress)
            chunk_vectors = self._client.embed(chunks)
            entity_vectors = self._client.embed(
                [f"{e.name}: {e.merged_description()}" if e.descriptions else e.name for e in draft.entities.values()]
            )
            self._store.add_document(doc_id, path.name, chunks, chunk_vectors, draft, entity_vectors)
        self._texts, self._matrix = self._store.load_chunks(doc_id)
        self._graph = self._store.load_graph(doc_id)
        self.doc_name = path.name
        logger.info(
            "Document ready: %s (%d chunks, %d entities, %d edges, fresh=%s)",
            path.name, len(self._texts), len(self._graph.names), self._graph.edge_count, fresh,
        )
        return fresh

    def _seed_entities(self, query: str, query_vec: np.ndarray) -> list[int]:
        """Pick entities by embedding similarity plus literal name mentions."""
        graph = self._graph
        if not graph.names:
            return []
        scores = graph.matrix @ query_vec
        k = min(_SEED_ENTITIES, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        seeds = top[np.argsort(-scores[top])].tolist()
        lowered = query.lower()
        for i, name in enumerate(graph.names):
            if i not in seeds and len(name) >= _MIN_NAME_MATCH_CHARS and name.lower() in lowered:
                seeds.append(i)
        return seeds

    def _graph_section(self, seeds: list[int], budget: int) -> str:
        """Build the graph facts and entity notes text for the seed entities."""
        graph = self._graph
        facts: list[str] = []
        notes: list[str] = []
        neighbours: set[int] = set()
        for seed in seeds:
            for line in graph.facts(seed, _MAX_FACTS_PER_SEED):
                if line not in facts:
                    facts.append(line)
            neighbours.update(n for n, _r, _o in graph.adjacency.get(seed, ())[:_MAX_FACTS_PER_SEED])
        for i in [*seeds, *sorted(neighbours - set(seeds))]:
            if graph.descriptions[i]:
                notes.append(f"{graph.names[i]}: {graph.descriptions[i]}")
        parts: list[str] = []
        used = 0
        for header, lines in (("Graph facts:", facts), ("Entities:", notes)):
            kept: list[str] = []
            for line in lines:
                if used + len(line) + 1 > budget:
                    break
                kept.append(line)
                used += len(line) + 1
            if kept:
                parts.append(header + "\n" + "\n".join(kept))
        return "\n\n".join(parts)

    def retrieve(self, query: str) -> str:
        """Assemble graph facts and relevant chunks into a compact context.

        Args:
            query: Search text.

        Returns:
            Context string (facts first, then document excerpts) within the character budget.

        Raises:
            RetrievalError: If no document is loaded.
        """
        if not self.ready:
            raise RetrievalError("No document loaded")
        query_vec = self._client.embed([query], is_query=True)[0]
        seeds = self._seed_entities(query, query_vec)

        scores = self._matrix @ query_vec
        k = min(self._settings.top_k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        chosen = top[np.argsort(-scores[top])].tolist()
        linked = {c for s in seeds for c in self._graph.entity_chunks.get(s, ())} - set(chosen)
        chosen += sorted(linked, key=lambda c: -scores[c])[:_EXTRA_LINKED_CHUNKS]
        logger.debug(
            "Seeds %s; chunks %s",
            [self._graph.names[s] for s in seeds],
            chosen,
        )

        graph_text = self._graph_section(seeds, int(_CONTEXT_CHARS * _FACTS_SHARE))
        remaining = _CONTEXT_CHARS - len(graph_text)
        excerpts: list[str] = []
        for c in chosen:
            if remaining <= 0:
                break
            piece = self._texts[c][:remaining]
            excerpts.append(piece)
            remaining -= len(piece) + 8
        sections = []
        if graph_text:
            sections.append(graph_text)
        if excerpts:
            sections.append("Document excerpts:\n" + "\n---\n".join(excerpts))
        return "\n\n".join(sections)


class Chatbot:
    """Interactive chat session supporting normal and document modes."""

    def __init__(self, settings: Settings, client: OllamaClient, engine: GraphRagEngine) -> None:
        """Create a session.

        Args:
            settings: Application settings.
            client: Ollama client.
            engine: GraphRAG engine.
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
        context = self._engine.retrieve(search)
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
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
        print(f"Indexing {path.name} (building knowledge graph, cached afterwards) ...", flush=True)

        def show_progress(done: int, total: int) -> None:
            print(f"\r  extracting entities: {done}/{total} chunks", end="" if done < total else "\n", flush=True)

        fresh = self._engine.load(path, show_progress)
        self.doc_mode = True
        self._history.clear()
        print(f"Document ready ({'indexed' if fresh else 'loaded from cache'}). Document mode ON.")
        print(self._engine.graph_summary().splitlines()[0])

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
        elif command == "/graph":
            if self._engine.ready:
                print(self._engine.graph_summary())
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
        print(f"GraphRAG chatbot ready in {mode}. Type /help for commands.")
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
                print(f"\n[error] {exc}")
            except KeyboardInterrupt:
                print("\n[interrupted]")


def main() -> int:
    """Program entry point.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="GraphRAG chatbot over a local document (Ollama).")
    parser.add_argument("document", nargs="?", help="path to a .txt, .md, .csv, .pdf or .docx file")
    args = parser.parse_args()

    store: GraphStore | None = None
    try:
        settings = get_settings()
        client = OllamaClient(settings)
        client.check_ready()
        store = GraphStore(settings.db_path)
        bot = Chatbot(settings, client, GraphRagEngine(settings, client, store))
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
