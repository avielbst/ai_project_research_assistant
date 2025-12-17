from __future__ import annotations

import os
from typing import List, Dict

from src.utils.config_loader import load_config
from src.rag.prompting import build_messages


class AnswerGenerator:
    """
    Generator with pluggable backends:
      - provider=ollama: OpenAI-compatible chat endpoint (your current setup)
      - provider=llama_cpp: local GGUF inference via llama-cpp-python (best for Docker/offline)

    Controlled via config.yml under `llm: ...`, with optional env overrides:
      - LLM_PROVIDER
      - LLM_BASE_URL
      - LLM_MODEL
      - LLM_MODEL_PATH
    """

    def __init__(self):
        cfg = load_config()
        llm = cfg.get("llm", {})

        self.provider = os.getenv("LLM_PROVIDER", llm.get("provider", "ollama")).strip().lower()
        self.temperature = float(llm.get("temperature", 0.2))
        self.max_tokens = int(llm.get("max_tokens", 400))

        # --- backend init ---
        if self.provider == "ollama":
            # OpenAI-compatible server (Ollama)
            from openai import OpenAI

            base_url = os.getenv("LLM_BASE_URL", llm.get("base_url", "http://localhost:11434/v1"))
            api_key = llm.get("api_key", "ollama")
            model = os.getenv("LLM_MODEL", llm.get("model", "qwen2.5:7b-instruct"))

            self.client = OpenAI(base_url=base_url, api_key=api_key)
            self.model = model
            self._backend = "ollama"

        elif self.provider == "llama_cpp":
            # Fully offline GGUF inference
            try:
                from llama_cpp import Llama
            except Exception as e:
                raise RuntimeError(
                    "llama-cpp-python is not installed. "
                    "Install it inside Docker (recommended) or in your local env if needed."
                ) from e

            model_path = os.getenv("LLM_MODEL_PATH", llm.get("model_path"))
            if not model_path:
                raise ValueError("Missing llm.model_path in config.yml (required for provider=llama_cpp).")

            # Make relative paths work from project root inside Docker
            # If your get_project_root() exists and you prefer it, you can swap this logic.
            model_path = os.path.normpath(model_path)

            n_ctx = int(llm.get("n_ctx", 4096))
            n_threads = int(llm.get("n_threads", max(os.cpu_count() or 4, 4)))
            n_gpu_layers = int(llm.get("n_gpu_layers", 0))
            stop = llm.get("stop", ["### User", "### System"])
            self.stop_sequences = stop if isinstance(stop, list) else [str(stop)]

            self.llama = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )
            self._backend = "llama_cpp"

        else:
            raise ValueError(f"Unsupported llm.provider='{self.provider}'. Use 'ollama' or 'llama_cpp'.")

    @staticmethod
    def _is_compliant(text: str) -> bool:
        # minimal check: citations + Sources used line
        t = (text or "")
        return ("Sources used:" in t) and ("[" in t and "]" in t)

    @staticmethod
    def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
        """
        Convert OpenAI-style messages to a single prompt for llama.cpp.
        Uses a simple, robust chat template with clear role headers.
        """
        parts: List[str] = []
        for m in messages:
            role = (m.get("role") or "").strip().lower()
            content = (m.get("content") or "").strip()

            if role == "system":
                parts.append("### System\n" + content)
            elif role == "user":
                parts.append("### User\n" + content)
            elif role == "assistant":
                parts.append("### Assistant\n" + content)
            else:
                # unknown role -> treat as user
                parts.append("### User\n" + content)

        # Ensure the model knows it should answer next
        parts.append("### Assistant\n")
        return "\n\n".join(parts)

    def generate(self, user_query: str, retrieved_context: str) -> str:
        messages = build_messages(user_query, retrieved_context)

        def call_ollama(temp: float, msgs: List[Dict[str, str]]) -> str:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=msgs,
                temperature=temp,
                max_tokens=self.max_tokens,
            )
            return resp.choices[0].message.content.strip()

        def call_llama_cpp(temp: float, msgs: List[Dict[str, str]]) -> str:
            prompt = self._messages_to_prompt(msgs)
            out = self.llama(
                prompt,
                temperature=temp,
                max_tokens=self.max_tokens,
                stop=self.stop_sequences,
            )
            return (out["choices"][0]["text"] or "").strip()

        # First attempt
        if self._backend == "ollama":
            out = call_ollama(self.temperature, messages)
        else:
            out = call_llama_cpp(self.temperature, messages)

        # One retry if format is violated
        if not self._is_compliant(out):
            messages_retry = messages + [{
                "role": "user",
                "content": (
                    "You did not follow the required format (citations + Sources used). "
                    "Rewrite and strictly follow the format."
                ),
            }]

            if self._backend == "ollama":
                out = call_ollama(0.0, messages_retry)
            else:
                out = call_llama_cpp(0.0, messages_retry)

        return out
