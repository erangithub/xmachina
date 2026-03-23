import os
from typing import Iterator
from openai import OpenAI
from xmachina import MessageEvent, Delta
from .base import LLM


class GroqLLM(LLM):
    def __init__(self, model: str = "llama-3.3-70b-versatile", api_key: str | None = None, **kwargs):
        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
            **kwargs,
        )

    def complete(self, messages: list[MessageEvent]) -> MessageEvent:
        from .openai import _to_dict
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[_to_dict(m) for m in messages],
        )
        msg = response.choices[0].message
        return MessageEvent(role=msg.role or "assistant", content=msg.content or "")

    def stream(self, messages: list[MessageEvent]) -> Iterator[Delta]:
        from .openai import _to_dict
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[_to_dict(m) for m in messages],
            stream=True,
        )
        for chunk in response:
            if delta := chunk.choices[0].delta.content:
                yield Delta(delta)
