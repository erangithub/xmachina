from typing import Iterator
from xmachina import MessageEvent, Delta
from .base import LLM


class EchoLLM(LLM):
    def complete(self, messages: list[MessageEvent]) -> MessageEvent:
        last_user = next((m for m in reversed(messages) if m.role == "user"), None)
        return MessageEvent(role="assistant", content=f"echo: {last_user.content if last_user else ''}")

    def stream(self, messages: list[MessageEvent]) -> Iterator[Delta]:
        last_user = next((m for m in reversed(messages) if m.role == "user"), None)
        for word in f"echo: {last_user.content if last_user else ''}".split():
            yield Delta(content=word + " ")
