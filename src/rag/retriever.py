from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import lancedb
from sentence_transformers import SentenceTransformer
import re
from src.utils.config_loader import load_config, get_project_root


@dataclass(frozen=True)
class RetrievedDoc:
    doc_idx: int
    doc_id: str
    title: str
    abstract: str
    distance: float


def _truncate(text: str, max_chars: int) -> str:
    text = (text or "").strip().replace("\n", " ")
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."

def truncate_by_sentences(text: str, max_chars: int) -> str:
    """
    Truncate text by full sentences without exceeding max_chars.
    """
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) <= max_chars:
        return text

    sentences = re.split(r"(?<=[.!?])\s+", text)

    out = []
    total = 0
    for s in sentences:
        if total + len(s) + 1 > max_chars:
            break
        out.append(s)
        total += len(s) + 1

    return " ".join(out).rstrip() + " …"

def _format_context(docs: List[RetrievedDoc], max_total_chars: int) -> str:
    blocks = []
    total = 0

    for i, d in enumerate(docs, start=1):
        block = (
            f"Document {i}\n"
            f"ID: {d.doc_id}\n"
            f"Title: {d.title}\n"
            f"Content:\n{d.abstract}"
        )

        if max_total_chars > 0 and total + len(block) + 2 > max_total_chars:
            break

        blocks.append(block)
        total += len(block) + 2

    return "\n\n---\n\n".join(blocks)



class Retriever:
    """
    Web-app friendly retriever:
    - loads config
    - opens LanceDB table
    - loads embedding model once
    """

    def __init__(self):
        self.project_root = get_project_root()
        self.cfg = load_config()

        vs = self.cfg["vector_store"]
        self.db_dir = self.project_root / vs["db_dir"]
        self.table_name = vs["table_name"]
        self.model_name = vs["embedding_model"]

        self.top_k = int(vs.get("top_k", 5))
        self.max_context_chars = int(vs.get("max_context_chars", 6000))
        self.max_abs_chars = int(vs.get("max_abstract_chars_per_doc", 1200))

        self.model = SentenceTransformer(self.model_name)
        self.db = lancedb.connect(str(self.db_dir))
        self.table = self.db.open_table(self.table_name)

    def retrieve(self, query: str, k: Optional[int] = None) -> Dict[str, Any]:
        k = self.top_k if k is None else int(k)

        qv = self.model.encode([query], normalize_embeddings=True).astype(np.float32)[0]

        rows = (
            self.table.search(qv)
            .limit(k)
            .select(["doc_idx", "id", "title", "abstract", "_distance"])
            .to_list()
        )

        docs: List[RetrievedDoc] = []
        for r in rows:
            docs.append(
                RetrievedDoc(
                    doc_idx=int(r["doc_idx"]),
                    doc_id=str(r["id"]),
                    title=str(r["title"]),
                    abstract=truncate_by_sentences(str(r["abstract"]), self.max_abs_chars),
                    distance=float(r["_distance"]),
                )
            )

        return {
            "retrieved_context": _format_context(docs, self.max_context_chars),
            "citations": [
                {"id": d.doc_id, "title": d.title, "distance": d.distance}
                for d in docs
            ],
        }
