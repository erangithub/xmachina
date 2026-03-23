from abc import ABC, abstractmethod
from typing import Iterator
from xmachina import MessageEvent, Delta


class LLM(ABC):
    @abstractmethod
    def complete(self, messages: list[MessageEvent]) -> MessageEvent:
        raise NotImplementedError

    def stream(self, messages: list[MessageEvent]) -> Iterator[Delta]:
        raise NotImplementedError("Streaming not supported")
