from __future__ import annotations

from openai import OpenAI

from src.utils.config_loader import load_config
from src.rag.prompting import build_messages


class AnswerGenerator:
    """
    Uses a local OpenAI-compatible server (Ollama).
    Loaded once; fast per-request.
    """

    def __init__(self):
        cfg = load_config()
        llm = cfg["llm"]

        self.client = OpenAI(
            base_url=llm["base_url"],
            api_key=llm.get("api_key", "ollama"),
        )
        self.model = llm["model"]
        self.temperature = float(llm.get("temperature", 0.2))
        self.max_tokens = int(llm.get("max_tokens", 400))

    @staticmethod
    def _is_compliant(text: str) -> bool:
        return ("Sources used:" in text) and ("[" in text and "]" in text)

    def generate(self, user_query: str, retrieved_context: str) -> str:
        messages = build_messages(user_query, retrieved_context)

        def call(temp: float) -> str:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                max_tokens=self.max_tokens,
            )
            return resp.choices[0].message.content.strip()

        out = call(self.temperature)

        # one automatic retry if format is violated
        if not self._is_compliant(out):
            # add a correction message (keeps context + forces compliance)
            messages_retry = messages + [{
                "role": "user",
                "content": (
                    "You did not follow the required format (citations + Sources used). "
                    "Rewrite and strictly follow the format."
                )
            }]
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages_retry,
                temperature=0.0,
                max_tokens=self.max_tokens,
            )
            out = resp.choices[0].message.content.strip()

        return out
