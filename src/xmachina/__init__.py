from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str          # JSON string — matches OpenAI standard
 
 
@dataclass(frozen=True)
class Message:
    role: str
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None     # for role == "tool"


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


    def messages(self) -> Iterator[Message]:
        stack, node = [], self.head
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
    history = list(log.messages())
    if window:
        history = history[-window:]
    context.extend(history)
    return context