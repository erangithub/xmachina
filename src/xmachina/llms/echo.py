from xmachina import Message
from .base import LLM


class EchoLLM(LLM):
    def complete(self, messages: list[Message]) -> Message:
        last = messages[-1] if messages else None
        return Message(role="assistant", content=f"Echo: {last.content if last else ''}")
