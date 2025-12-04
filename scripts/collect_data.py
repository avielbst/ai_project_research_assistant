import json
from pathlib import Path
from typing import Iterable, Dict

from src.utils.config_loader import load_config, get_project_root
from src.utils.dataset_utils import fetch_papers_weighted


def save_jsonl(
    records: Iterable[Dict],
    file_path: Path,
    mode: str = "w",
    flush_every: int = 100,
) -> None:
    """
    Write extracted records into a JSONL file in a streaming-friendly way.
    Buffers lines and writes every `flush_every` records.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open(mode, encoding="utf-8") as f:
        buffer = []
        for i, rec in enumerate(records, 1):
            buffer.append(json.dumps(rec, ensure_ascii=False))
            if i % flush_every == 0:
                f.write("\n".join(buffer) + "\n")
                buffer.clear()

        if buffer:
            f.write("\n".join(buffer) + "\n")


def main():
    cfg = load_config()
    project_root = get_project_root()

    rel_save_path = cfg["dataset"]["save_path"]
    save_path = project_root / rel_save_path
    flush_every = cfg["dataset"]["flush_every"]

    records_iter = fetch_papers_weighted(cfg)

    save_jsonl(records_iter, save_path, mode="w", flush_every=flush_every)
    print(f"Finished writing dataset to: {save_path}")


if __name__ == "__main__":
    main()
