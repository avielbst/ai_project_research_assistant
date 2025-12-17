from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI

from src.rag.retriever import Retriever
from src.rag.generator import AnswerGenerator


@asynccontextmanager
async def lifespan(app: FastAPI):
    retriever = Retriever()
    generator = AnswerGenerator()
    app.state.retriever = retriever
    app.state.generator = generator

    try:
        yield
    finally:
        pass
