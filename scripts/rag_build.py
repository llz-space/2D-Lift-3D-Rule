#!/usr/bin/env python3
"""Build a lightweight lexical RAG index from rag/corpus markdown files."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "rag" / "corpus"
INDEX_PATH = ROOT / "rag" / "index.json"

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def main() -> None:
    files = sorted(CORPUS_DIR.glob("*.md"))
    if not files:
        raise SystemExit(f"No corpus files found in {CORPUS_DIR}")

    documents = []
    doc_term_counts: list[Counter[str]] = []
    df = defaultdict(int)

    for idx, path in enumerate(files):
        text = path.read_text(encoding="utf-8")
        tokens = tokenize(text)
        term_counts = Counter(tokens)
        for term in term_counts:
            df[term] += 1
        documents.append(
            {
                "doc_id": idx,
                "path": str(path.relative_to(ROOT)),
                "title": text.splitlines()[0].lstrip("# ").strip() if text else path.stem,
                "text": text,
                "length": len(tokens),
            }
        )
        doc_term_counts.append(term_counts)

    n_docs = len(documents)
    idf = {term: math.log((n_docs + 1) / (freq + 1)) + 1.0 for term, freq in df.items()}

    inverted = defaultdict(list)
    for doc in documents:
        doc_id = doc["doc_id"]
        tf = doc_term_counts[doc_id]
        norm = sum(v * v for v in tf.values()) ** 0.5 or 1.0
        for term, cnt in tf.items():
            weight = (cnt / norm) * idf[term]
            inverted[term].append([doc_id, round(weight, 8)])

    index = {
        "meta": {
            "num_docs": n_docs,
            "tokenizer": "regex:[A-Za-z0-9_]+|[\\u4e00-\\u9fff]",
            "corpus_dir": str(CORPUS_DIR.relative_to(ROOT)),
        },
        "documents": documents,
        "idf": idf,
        "inverted_index": dict(inverted),
    }

    INDEX_PATH.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Built index with {n_docs} docs -> {INDEX_PATH}")


if __name__ == "__main__":
    main()
