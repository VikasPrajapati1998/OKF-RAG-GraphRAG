"""Document loading, hashing and chunking."""

import csv
import hashlib
import io
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from exception import DocumentError
from logger import logger

SUPPORTED_EXTENSIONS = (".txt", ".md", ".csv", ".pdf", ".docx")

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_MC = "{http://schemas.openxmlformats.org/markup-compatibility/2006}"
_SKIPPED_INLINE = {"pPr", "rPr", "del", "delText", "instrText", "Fallback"}


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file.

    Args:
        path: File to hash.

    Returns:
        Hex digest string.

    Raises:
        DocumentError: If the file cannot be read.
    """
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise DocumentError(f"Cannot read {path}: {exc}") from exc
    return digest.hexdigest()


def load_document(path: Path) -> str:
    """Extract plain text from a supported document.

    Args:
        path: Path to a ``.txt``, ``.md``, ``.csv``, ``.pdf`` or ``.docx`` file.

    Returns:
        The extracted text.

    Raises:
        DocumentError: If the file is missing, unsupported, unreadable or empty.
    """
    if not path.is_file():
        raise DocumentError(f"File not found: {path}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise DocumentError(f"Unsupported file type '{suffix}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}")

    try:
        if suffix == ".pdf":
            text = _read_pdf(path)
        elif suffix == ".docx":
            text = _read_docx(path)
        elif suffix == ".csv":
            text = _read_csv(path)
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
    except DocumentError:
        raise
    except Exception as exc:  # parser libraries raise many unrelated exception types
        raise DocumentError(f"Failed to parse {path.name}: {exc}") from exc

    text = re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n"))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        raise DocumentError(f"No extractable text in {path.name}")
    logger.info("Loaded %s (%d characters)", path.name, len(text))
    return text


def _read_csv(path: Path) -> str:
    """Render a CSV file as one ``column: value | column: value`` line per row.

    The delimiter is auto-detected and the first row is treated as the header.

    Args:
        path: CSV file path.

    Returns:
        One line per data row, so each value stays tied to its column name.

    Raises:
        DocumentError: If the file cannot be decoded or parsed as CSV.
    """
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise DocumentError(f"Cannot read {path}: {exc}") from exc
    try:
        content = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        content = raw.decode("cp1252", errors="replace")

    try:
        dialect = csv.Sniffer().sniff(content[:8192], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    try:
        rows = [row for row in csv.reader(io.StringIO(content), dialect) if any(c.strip() for c in row)]
    except csv.Error as exc:
        raise DocumentError(f"Invalid CSV {path.name}: {exc}") from exc
    if not rows:
        return ""

    header = [h.strip() or f"column_{i + 1}" for i, h in enumerate(rows[0])]
    lines = [f"Columns: {', '.join(header)}"]
    for row in rows[1:]:
        cells = [
            f"{header[i] if i < len(header) else f'column_{i + 1}'}: {value.strip()}"
            for i, value in enumerate(row)
            if value.strip()
        ]
        if cells:
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def _read_pdf(path: Path) -> str:
    """Read all pages of a PDF."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise DocumentError("PDF support requires 'pypdf' (pip install pypdf)") from exc
    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def _read_docx(path: Path) -> str:
    """Read all text of a DOCX file directly from its OOXML, ignoring images.

    Covers body paragraphs and tables in document order (headings as Markdown,
    list items as bullets), content controls, hyperlinks, text boxes and
    shapes, equations, tracked insertions (deleted text is dropped), headers,
    footers, footnotes, endnotes and comments.

    Args:
        path: DOCX file path.

    Returns:
        The extracted text.

    Raises:
        DocumentError: If the file is not a valid DOCX.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            names = sorted(archive.namelist())
            if "word/document.xml" not in names:
                raise DocumentError(f"{path.name} is not a valid .docx file (no word/document.xml)")
            body = ElementTree.fromstring(archive.read("word/document.xml")).find(f"{_W}body")
            blocks = _docx_blocks(body) if body is not None else []

            extras: list[str] = []
            for name in names:
                if re.fullmatch(r"word/(header|footer)\d*\.xml", name):
                    text = "\n".join(_docx_blocks(ElementTree.fromstring(archive.read(name))))
                    if text and text not in extras:
                        extras.append(text)
            for name, label, tag in (
                ("word/footnotes.xml", "Footnote", "footnote"),
                ("word/endnotes.xml", "Endnote", "endnote"),
                ("word/comments.xml", "Comment", "comment"),
            ):
                if name in names:
                    extras.extend(_docx_notes(ElementTree.fromstring(archive.read(name)), label, tag))
    except (zipfile.BadZipFile, ElementTree.ParseError, KeyError) as exc:
        raise DocumentError(f"{path.name} is not a valid .docx file: {exc}") from exc
    return "\n\n".join(blocks + extras)


def _local(element: ElementTree.Element) -> str:
    """Return an element's tag without its XML namespace."""
    return element.tag.rsplit("}", 1)[-1]


def _docx_blocks(container: ElementTree.Element) -> list[str]:
    """Convert the paragraphs and tables under a container, in order."""
    blocks: list[str] = []
    for child in container:
        tag = _local(child)
        if tag == "p":
            blocks.extend(_docx_paragraph(child))
        elif tag == "tbl":
            table = _docx_table(child)
            if table:
                blocks.append(table)
        elif tag == "sdt":
            content = child.find(f"{_W}sdtContent")
            if content is not None:
                blocks.extend(_docx_blocks(content))
        elif tag in ("ins", "moveTo", "customXml", "smartTag"):
            blocks.extend(_docx_blocks(child))
    return blocks


def _docx_paragraph(paragraph: ElementTree.Element) -> list[str]:
    """Return a paragraph's text (with heading/list markup) plus any text-box blocks."""
    pieces: list[str] = []
    boxes: list[str] = []
    _docx_collect(paragraph, pieces, boxes)
    text = "".join(pieces).strip()
    blocks: list[str] = []
    if text:
        properties = paragraph.find(f"{_W}pPr")
        style = ""
        is_list = False
        if properties is not None:
            style_el = properties.find(f"{_W}pStyle")
            style = style_el.get(f"{_W}val", "") if style_el is not None else ""
            is_list = properties.find(f"{_W}numPr") is not None
        heading = re.fullmatch(r"(?i)heading\s*(\d)", style)
        if style.lower() == "title":
            text = f"# {text}"
        elif heading:
            text = f"{'#' * int(heading.group(1))} {text}"
        elif is_list or style.lower().startswith("list"):
            text = f"- {text}"
        blocks.append(text)
    blocks.extend(boxes)
    return blocks


def _docx_collect(element: ElementTree.Element, pieces: list[str], boxes: list[str]) -> None:
    """Recursively gather inline text; text boxes go to ``boxes`` to avoid duplication."""
    for child in element:
        tag = _local(child)
        if tag in _SKIPPED_INLINE:
            continue
        if tag == "t":  # w:t and equation m:t
            pieces.append(child.text or "")
        elif tag == "tab":
            pieces.append("\t")
        elif tag in ("br", "cr"):
            pieces.append("\n")
        elif tag == "noBreakHyphen":
            pieces.append("-")
        elif tag == "txbxContent":
            boxes.extend(_docx_blocks(child))
        elif child.tag == f"{_MC}AlternateContent":
            choice = child.find(f"{_MC}Choice")  # Fallback duplicates the Choice content
            if choice is not None:
                _docx_collect(choice, pieces, boxes)
        else:
            _docx_collect(child, pieces, boxes)


def _docx_table(table: ElementTree.Element) -> str:
    """Render a table as ``cell | cell`` rows; nested tables are flattened with ``;``."""
    rows: list[str] = []
    for row in table:
        if _local(row) != "tr":
            continue
        cells = [
            " ".join(b.replace("\n", "; ") for b in _docx_blocks(cell))
            for cell in row
            if _local(cell) == "tc"
        ]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _docx_notes(root: ElementTree.Element, label: str, tag: str) -> list[str]:
    """Return footnotes, endnotes or comments as ``Label: text`` strings."""
    notes: list[str] = []
    for note in root.iter(f"{_W}{tag}"):
        if note.get(f"{_W}type") in ("separator", "continuationSeparator", "continuationNotice"):
            continue
        text = " ".join(_docx_blocks(note)).strip()
        if text:
            notes.append(f"{label}: {text}")
    return notes


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into paragraph/sentence-aware overlapping chunks.

    Args:
        text: Text to split.
        chunk_size: Target maximum characters per chunk.
        overlap: Characters carried over from the previous chunk.

    Returns:
        Non-empty chunks in document order.
    """
    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= chunk_size:
            units.append(paragraph)
        else:
            units.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", paragraph) if s.strip())

    chunks: list[str] = []
    current = ""
    for unit in units:
        while len(unit) > chunk_size:  # hard-split pathological long sentences
            if current:
                chunks.append(current)
                current = ""
            chunks.append(unit[:chunk_size])
            unit = unit[chunk_size - overlap :]
        if current and len(current) + len(unit) + 1 > chunk_size:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail} {unit}".strip() if tail else unit
        else:
            current = f"{current}\n{unit}".strip() if current else unit
    if current:
        chunks.append(current)
    return chunks
