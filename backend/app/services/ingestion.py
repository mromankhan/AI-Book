import os
import re
import hashlib
from pathlib import Path
from typing import List

import tiktoken

from app.config import settings
from app.services.embeddings import get_embeddings_batch
from app.services.qdrant_service import ensure_collection, upsert_chunks


def count_tokens(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def chunk_text(text: str, max_tokens: int = 512, overlap_tokens: int = 50) -> List[str]:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)

    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        if chunk_text.strip():
            chunks.append(chunk_text.strip())
        start = end - overlap_tokens

    return chunks


def parse_markdown_file(filepath: str) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract frontmatter title if present
    title_match = re.search(r"^---\s*\n.*?title:\s*['\"]?(.+?)['\"]?\s*\n.*?---", content, re.DOTALL)
    title = title_match.group(1) if title_match else Path(filepath).stem

    # Remove frontmatter
    content = re.sub(r"^---\s*\n.*?---\s*\n", "", content, flags=re.DOTALL)

    # Extract sections by headings
    sections = []
    current_section = "Introduction"
    current_content = []

    for line in content.split("\n"):
        heading_match = re.match(r"^(#{1,3})\s+(.+)$", line)
        if heading_match:
            if current_content:
                text = "\n".join(current_content).strip()
                if text:
                    sections.append({"section": current_section, "content": text})
            current_section = heading_match.group(2).strip()
            current_content = []
        else:
            current_content.append(line)

    # Add last section
    if current_content:
        text = "\n".join(current_content).strip()
        if text:
            sections.append({"section": current_section, "content": text})

    return {"title": title, "sections": sections}


def ingest_book_content(docs_dir: str):
    ensure_collection()

    docs_path = Path(docs_dir)
    md_files = sorted(docs_path.rglob("*.md"))

    all_chunks = []
    total_files = 0

    for md_file in md_files:
        # Skip category files and non-content files
        if md_file.name.startswith("_") or md_file.name == "index.md":
            continue

        relative_path = md_file.relative_to(docs_path)
        chapter = str(relative_path.parent / relative_path.stem)

        parsed = parse_markdown_file(str(md_file))
        total_files += 1

        for section_data in parsed["sections"]:
            section_text = section_data["content"]
            section_name = section_data["section"]

            # Skip very short sections
            if count_tokens(section_text) < 20:
                continue

            text_chunks = chunk_text(
                section_text,
                max_tokens=settings.chunk_size,
                overlap_tokens=settings.chunk_overlap,
            )

            for i, chunk in enumerate(text_chunks):
                chunk_id = hashlib.md5(f"{chapter}:{section_name}:{i}".encode()).hexdigest()
                all_chunks.append({
                    "chunk_id": chunk_id,
                    "chapter": chapter,
                    "section": section_name,
                    "content": chunk,
                })

    # Batch embed and upsert
    batch_size = 50
    total_chunks = len(all_chunks)

    for i in range(0, total_chunks, batch_size):
        batch = all_chunks[i:i + batch_size]
        texts = [c["content"] for c in batch]
        embeddings = get_embeddings_batch(texts)
        upsert_chunks(batch, embeddings)
        print(f"  Ingested {min(i + batch_size, total_chunks)}/{total_chunks} chunks...")

    print(f"Ingestion complete: {total_files} files, {total_chunks} chunks indexed.")
    return {"files": total_files, "chunks": total_chunks}
