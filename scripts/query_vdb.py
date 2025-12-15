#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import lancedb
from sentence_transformers import SentenceTransformer

from src.utils.config_loader import load_config, get_project_root


def main() -> None:
    project_root = get_project_root()
    cfg = load_config()

    db_dir = project_root / cfg["vector_store"]["db_dir"]
    table_name = cfg["vector_store"]["table_name"]
    model_name = cfg["vector_store"]["embedding_model"]
    default_k = int(cfg["vector_store"].get("top_k", 5))

    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="User query text")
    parser.add_argument("-k", "--top_k", type=int, default=None, help="Override top_k from config")
    parser.add_argument("--show-abstract", action="store_true", help="Also print abstracts (truncated)")
    args = parser.parse_args()

    k = args.top_k if args.top_k is not None else default_k

    if not db_dir.exists():
        raise FileNotFoundError(f"Vector DB directory not found: {db_dir}. Did you run build_index.py?")

    model = SentenceTransformer(model_name)
    db = lancedb.connect(str(db_dir))
    table = db.open_table(table_name)

    qv = model.encode([args.query], normalize_embeddings=True).astype(np.float32)[0]

    cols = ["doc_idx", "id", "title", "_distance"]
    if args.show_abstract:
        cols.append("abstract")

    results = table.search(qv).limit(k).select(cols).to_list()

    print(f"\nQuery: {args.query}\nTop-{k} results:\n")
    for i, r in enumerate(results, start=1):
        print(f"{i}. [{r.get('id')}] {r.get('title')}  (distance={r.get('_distance'):.4f})")
        if args.show_abstract:
            abs_text = (r.get("abstract") or "").strip().replace("\n", " ")
            if len(abs_text) > 600:
                abs_text = abs_text[:600] + "…"
            print(f"   abstract: {abs_text}")
        print()


if __name__ == "__main__":
    main()
