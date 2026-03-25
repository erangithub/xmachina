from typing import Iterator
from xmachina import Message, Delta
from .base import LLM
from time import sleep

class EchoLLM(LLM):
    def complete(self, messages: list[Message]) -> Message:
        last_user = next((m for m in reversed(messages) if m.role == "user"), None)
        return Message(role="assistant", content=f"echo: {last_user.content if last_user else ''}")

    def stream(self, messages: list[Message]) -> Iterator[Delta]:
        last_user = next((m for m in reversed(messages) if m.role == "user"), None)
        for word in f"echo: {last_user.content if last_user else ''}".split():
            sleep(0.1)
            yield Delta(content=word + " ")

