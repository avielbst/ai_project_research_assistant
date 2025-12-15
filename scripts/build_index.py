import os
import json
from pathlib import Path
import yaml
import numpy as np
import lancedb
from sentence_transformers import SentenceTransformer
from src.utils.config_loader import load_config, get_project_root

project_root = get_project_root()
cfg = load_config()

DATA_PATH = project_root / cfg["dataset"]["save_path"]
BATCH_SIZE = int(cfg["dataset"]["batch_size"])

DB_DIR = project_root / cfg["vector_store"]["db_dir"]
TABLE = cfg["vector_store"]["table_name"]
MODEL_NAME = cfg["vector_store"]["embedding_model"]



def stream_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"DATA_PATH not found: {DATA_PATH}")

    DB_DIR.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(MODEL_NAME)
    db = lancedb.connect(str(DB_DIR))

    # Clean rebuild for now (simple + deterministic)
    try:
        db.drop_table(TABLE)
    except Exception:
        pass

    table = None
    batch = []
    doc_idx = 0
    kept = 0
    skipped = 0

    def flush():
        nonlocal table, batch, kept
        if not batch:
            return

        texts = [r["abstract"] for r in batch]
        vecs = model.encode(texts, normalize_embeddings=True)
        vecs = vecs.astype(np.float32)

        for r, v in zip(batch, vecs):
            r["vector"] = v

        if table is None:
            table = db.create_table(TABLE, data=batch)
        else:
            table.add(batch)

        kept += len(batch)
        batch = []

    for rec in stream_jsonl(DATA_PATH):
        rid = str(rec.get("id", "")).strip()
        title = str(rec.get("title", "")).strip()
        abstract = str(rec.get("abstract", "")).strip()

        if not rid or not title or not abstract:
            skipped += 1
            continue

        batch.append(
            {
                "doc_idx": doc_idx,
                "id": rid,
                "title": title,
                "abstract": abstract,
            }
        )
        doc_idx += 1

        if len(batch) >= BATCH_SIZE:
            flush()

    flush()

    if table is None:
        raise RuntimeError("No valid documents were indexed.")

    print(f"✅ Indexed {kept} documents into {DB_DIR} / table '{TABLE}' (skipped {skipped})")


if __name__ == "__main__":
    main()
