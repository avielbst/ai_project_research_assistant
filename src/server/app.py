from __future__ import annotations

import os
from typing import Any, List, Set, Dict, Optional
import re

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.server.startup import lifespan
from src.rag.retriever import Retriever
from src.rag.generator import AnswerGenerator

CITATION_RE = re.compile(r"\[(\d{4}\.\d{5}v\d+)\]")  # e.g. [2510.02964v1]
DOC_SPLIT_RE = re.compile(r"\n---\n")               # matches your _format_context separator
DOC_ID_RE = re.compile(r"^DOC\s+\[(.*?)\]\s*$", re.MULTILINE)
app = FastAPI(title="Research Assistant - Aviel", lifespan=lifespan)

def extract_used_ids(answer: str) -> List[str]:
    # preserve order, unique
    seen: Set[str] = set()
    out: List[str] = []
    for m in CITATION_RE.finditer(answer or ""):
        cid = m.group(1)
        if cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out

def filter_citations(all_citations: List[Dict], used_ids: List[str]) -> List[Dict]:
    used = set(used_ids)
    return [c for c in all_citations if c.get("doc_id") in used]

def filter_context_by_ids(retrieved_context: str, used_ids: List[str]) -> List[str]:
    """
    Returns list of DOC blocks matching used_ids, preserving used_ids order when possible.
    """
    if not retrieved_context:
        return []

    blocks = DOC_SPLIT_RE.split(retrieved_context)
    by_id: Dict[str, str] = {}
    for b in blocks:
        m = DOC_ID_RE.search(b.strip())
        if m:
            by_id[m.group(1)] = b.strip()

    # order by used_ids; keep only those found
    return [by_id[cid] for cid in used_ids if cid in by_id]

class AnswerRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    debug: bool = False


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    static_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(static_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.post("/answer")
async def answer(req: AnswerRequest) -> Dict[str, Any]:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    retriever: Retriever = app.state.retriever
    generator: AnswerGenerator = app.state.generator

    def run_rag():
        r = retriever.retrieve(req.query, k=req.top_k) if req.top_k is not None else retriever.retrieve(req.query)
        ans = generator.generate(req.query, r["retrieved_context"])
        return r, ans

    r, ans = await run_in_threadpool(run_rag)

    all_citations = [
        {
            "doc_id": c.get("id"),
            "title": c.get("title"),
            "distance": c.get("distance"),
            "rerank_score": c.get("rerank_score"),
            "url": c.get("url"),
        }
        for c in r.get("citations", [])
    ]

    used_ids = extract_used_ids(ans)

    # If model didn't cite anything, fall back to showing retrieved (debug-friendly)
    if used_ids:
        citations = filter_citations(all_citations, used_ids)
        ctx_blocks = filter_context_by_ids(r.get("retrieved_context", ""), used_ids)
    else:
        citations = all_citations
        ctx_blocks = [r.get("retrieved_context", "")] if r.get("retrieved_context") else []

    payload: Dict[str, Any] = {
        "answer": ans,
        "citations": citations,
        "retrieved_context": ctx_blocks,
        "used_ids": used_ids,  # optional, helps debugging/UI
    }

    if req.debug:
        payload["debug"] = {"query": req.query, "top_k": req.top_k}

    return payload
