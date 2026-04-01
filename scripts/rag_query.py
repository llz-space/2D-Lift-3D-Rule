#!/usr/bin/env python3
"""Query the local RAG index and print top-k matched documents."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / "rag" / "index.json"
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def load_index() -> dict:
    if not INDEX_PATH.exists():
        raise SystemExit(f"Index not found: {INDEX_PATH}. Run: python scripts/rag_build.py")
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


def search(index: dict, query: str, top_k: int) -> list[tuple[float, dict]]:
    tokens = tokenize(query)
    if not tokens:
        return []

    q_tf = Counter(tokens)
    q_norm = sum(v * v for v in q_tf.values()) ** 0.5 or 1.0
    idf = index["idf"]
    inv = index["inverted_index"]

    scores = defaultdict(float)
    for term, cnt in q_tf.items():
        q_weight = (cnt / q_norm) * idf.get(term, math.log((index["meta"]["num_docs"] + 1)) + 1.0)
        for doc_id, d_weight in inv.get(term, []):
            scores[doc_id] += q_weight * d_weight

    docs = {doc["doc_id"]: doc for doc in index["documents"]}
    ranked = sorted(((score, docs[doc_id]) for doc_id, score in scores.items()), reverse=True)
    return ranked[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="query text")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    index = load_index()
    results = search(index, args.query, args.top_k)
    if not results:
        print("No results.")
        return

    for rank, (score, doc) in enumerate(results, start=1):
        print(f"[{rank}] score={score:.4f} path={doc['path']} title={doc['title']}")


if __name__ == "__main__":
    main()
