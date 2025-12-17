from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import lancedb
from sentence_transformers import SentenceTransformer, CrossEncoder
import re
from src.utils.config_loader import load_config, get_project_root


@dataclass(frozen=True)
class RetrievedDoc:
    doc_idx: int
    doc_id: str
    title: str
    abstract: str
    distance: float
    rerank_score: Optional[float] = None

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

def _format_context(docs, max_chars: int) -> str:
    chunks = []
    total = 0
    for d in docs:
        block = (
            f"DOC [{d.doc_id}]\n"
            f"Title: {d.title}\n"
            f"Abstract: {d.abstract}\n"
        )
        if total + len(block) > max_chars:
            break
        chunks.append(block)
        total += len(block)
    return "\n---\n".join(chunks)

class Retriever:
    def __init__(self):
        self.project_root = get_project_root()
        self.cfg = load_config()

        vs = self.cfg["vector_store"]
        self.db_dir = self.project_root / vs["db_dir"]
        self.table_name = vs["table_name"]
        self.model_name = vs["embedding_model"]

        self.top_k = int(vs["top_k"])
        self.initial_k = int(vs["initial_retrieval_k"])
        self.max_context_chars = int(vs["max_context_chars"])
        self.max_abs_chars = int(vs["max_abstract_chars_per_doc"])

        # reranker config
        self.use_reranker = bool(vs["use_reranker"])
        self.reranker_model_name = vs["reranker_model"]
        self.reranker_max_length = int(vs["reranker_max_length"])

        self.model = SentenceTransformer(self.model_name)
        self.db = lancedb.connect(str(self.db_dir))
        self.table = self.db.open_table(self.table_name)

        self.reranker = None
        if self.use_reranker:
            self.reranker = CrossEncoder(self.reranker_model_name, max_length=self.reranker_max_length)

    def _dedup_by_id_keep_best_distance(self, docs: List[RetrievedDoc]) -> List[RetrievedDoc]:
        best: dict[str, RetrievedDoc] = {}
        for d in docs:
            prev = best.get(d.doc_id)
            if prev is None or d.distance < prev.distance:
                best[d.doc_id] = d
        return list(best.values())

    def _rerank(self, query: str, docs: List[RetrievedDoc]) -> List[RetrievedDoc]:
        if not self.reranker or not docs:
            return docs

        pairs = [(f"Question: {query}", f"Paper title: {d.title}\nPaper abstract: {d.abstract}") for d in docs]

        scores = self.reranker.predict(pairs)

        rescored = [
            RetrievedDoc(
                doc_idx=d.doc_idx,
                doc_id=d.doc_id,
                title=d.title,
                abstract=d.abstract,
                distance=d.distance,
                rerank_score=float(s),
            )
            for d, s in zip(docs, scores)
        ]

        rescored.sort(key=lambda x: x.rerank_score if x.rerank_score is not None else -1e9, reverse=True)
        return rescored

    def retrieve(self, query: str, k: Optional[int] = None) -> Dict[str, Any]:
        k = self.top_k if k is None else int(k)
        initial_k = max(self.initial_k, k)

        qv = self.model.encode([query], normalize_embeddings=True).astype(np.float32)[0]

        rows = (
            self.table.search(qv)
            .limit(initial_k)
            .select(["doc_idx", "id", "title", "abstract", "_distance"])
            .to_list()
        )

        candidates: List[RetrievedDoc] = []
        for r in rows:
            candidates.append(
                RetrievedDoc(
                    doc_idx=int(r["doc_idx"]),
                    doc_id=str(r["id"]),
                    title=str(r["title"]),
                    abstract=truncate_by_sentences(str(r["abstract"]), self.max_abs_chars),
                    distance=float(r["_distance"]),
                )
            )

        candidates = self._dedup_by_id_keep_best_distance(candidates)

        ranked = self._rerank(query, candidates)
        final_docs = ranked[:k]

        citations = []
        seen = set()
        for d in final_docs:
            if d.doc_id in seen:
                continue
            seen.add(d.doc_id)
            citations.append({
                "id": d.doc_id,
                "title": d.title,
                "distance": d.distance,
                "rerank_score": d.rerank_score
            })

        return {
            "retrieved_context": _format_context(final_docs, self.max_context_chars),
            "citations": citations,
            "ranked": ranked,
        }
