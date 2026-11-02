"""Document parsing and chunking with metadata preservation.

Splits documents into meaningful passages while preserving document and
section metadata. All retrieved text is treated as untrusted data.
"""

import re
from typing import List, Optional

from .config import AgentConfig
from .models import Passage, SourceRecord


def clean_text(text: str) -> str:
    """Basic text cleaning: normalize whitespace, remove control characters."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text.strip()


def extract_sections(html_or_text: str) -> List[dict]:
    """Split a document into sections, preserving section headings.

    Returns a list of dicts with keys: section, text.
    If no headings are found, returns a single section with the full text.
    """
    # Try to find common section heading patterns
    heading_pattern = re.compile(r"(?m)^(#{1,6}\s+.+|[A-Z][A-Z\s]{3,}|^\d+\.\s+.+)$", re.MULTILINE)
    lines = html_or_text.split("\n")
    sections: List[dict] = []
    current_section = "General"
    current_text: List[str] = []

    for line in lines:
        if heading_pattern.match(line.strip()):
            if current_text:
                sections.append({"section": current_section, "text": "\n".join(current_text)})
                current_text = []
            current_section = line.strip().lstrip("#").strip()
        else:
            current_text.append(line)

    if current_text:
        sections.append({"section": current_section, "text": "\n".join(current_text)})

    if not sections:
        sections = [{"section": "General", "text": html_or_text}]

    return sections


def chunk_text(text: str, max_chunk_chars: int = 1200, overlap_chars: int = 150) -> List[str]:
    """Split text into overlapping chunks of roughly max_chunk_chars characters.

    Overlap helps preserve context across chunk boundaries.
    """
    text = clean_text(text)
    if len(text) <= max_chunk_chars:
        return [text] if text else []

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chunk_chars, len(text))
        # Try to break at a sentence boundary
        if end < len(text):
            sentence_end = text.rfind(". ", start, end)
            if sentence_end != -1 and sentence_end > start:
                end = sentence_end + 2
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap_chars
        if start < 0:
            start = 0

    return chunks


def document_to_passages(
    source: SourceRecord,
    full_text: str,
    max_chunk_chars: int = 1200,
    overlap_chars: int = 150,
) -> List[Passage]:
    """Convert a source document into a list of Passage objects.

    Each passage preserves source URL, title, section, chunk index, and metadata.
    """
    sections = extract_sections(full_text)
    passages: List[Passage] = []
    global_index = 0

    for sec in sections:
        section_text = clean_text(sec["text"])
        if not section_text:
            continue
        chunks = chunk_text(section_text, max_chunk_chars=max_chunk_chars, overlap_chars=overlap_chars)
        for chunk in chunks:
            passages.append(
                Passage(
                    text=chunk,
                    source_url=source.url,
                    source_title=source.title,
                    section=sec["section"],
                    chunk_index=global_index,
                    metadata={
                        "source_type": source.source_type,
                        "publication_date": source.publication_date,
                        "retrieval_time": source.retrieval_time,
                        "discovered_by": source.discovered_by,
                    },
                )
            )
            global_index += 1

    return passages