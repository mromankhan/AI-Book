#!/usr/bin/env python3
"""CLI script to ingest book content into Qdrant vector store."""

import sys
import os
from pathlib import Path

# Add backend root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.services.ingestion import ingest_book_content


def main():
    # Default to Book/docs relative to project root
    project_root = Path(__file__).parent.parent.parent
    docs_dir = project_root / "Book" / "docs"

    if len(sys.argv) > 1:
        docs_dir = Path(sys.argv[1])

    if not docs_dir.exists():
        print(f"Error: docs directory not found at {docs_dir}")
        sys.exit(1)

    print(f"Ingesting book content from: {docs_dir}")
    print("=" * 60)

    result = ingest_book_content(str(docs_dir))

    print("=" * 60)
    print(f"Done! Processed {result['files']} files, {result['chunks']} chunks indexed.")


if __name__ == "__main__":
    main()
