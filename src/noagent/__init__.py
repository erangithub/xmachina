from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True)
class EventNode:
    message: Message
    parent: "EventNode | None"


@dataclass(frozen=True)
class Conversation:
    head: EventNode | None = None

    def append(self, msg: Message) -> "Conversation":
        node = EventNode(
            message=msg,
            parent=self.head,
        )
        return Conversation(head=node)

    @staticmethod
    def start(*messages: Message) -> "Conversation":
        log = Conversation()
        for msg in messages:
            log = log.append(msg)
        return log


def events(log: Conversation) -> Iterator[Message]:
    stack, node = [], log.head
    while node:
        stack.append(node.message)
        node = node.parent
    while stack:
        yield stack.pop()


def build_context(
    log: Conversation,
    system: str | None = None,
    injections: list[str] | None = None,
    window: int | None = None,
) -> list[Message]:
    context = []
    if system:
        context.append(Message("system", system))
    for text in (injections or []):
        context.append(Message("system", text))
    history = list(events(log))
    if window:
        history = history[-window:]
    context.extend(history)
    return context


class EchoLLM:
    def complete(self, context: list[Message]) -> Message:
        last_user = next(m for m in reversed(context) if m.role == "user")
        return Message("assistant", f"echo: {last_user.content}")
