from abc import ABC, abstractmethod
from typing import Iterator
from xmachina import Message, Delta


class LLM(ABC):
    @abstractmethod
    def complete(self, messages: list[Message]) -> Message:
        raise NotImplementedError

    def stream(self, messages: list[Message]) -> Iterator[Delta]:
        raise NotImplementedError("Streaming not supported")
