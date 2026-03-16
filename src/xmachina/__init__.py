from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator


# A tool call requested by the LLM — matches OpenAI standard
@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str          # JSON string


# A complete event in the conversation — lives in the log permanently
@dataclass(frozen=True)
class Message:
    role: str               # 'user', 'assistant', 'system', 'tool'
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None     # for role == "tool"


# A fragment in flight during streaming — never touches the log
@dataclass(frozen=True)
class Delta:
    content: str


# A node in the conversation tree — points to its parent
@dataclass(frozen=True)
class EventNode:
    message: Message
    parent: EventNode | None
    depth: int              # O(1) length — depth of this node from root


# The conversation — an immutable tree, O(1) branching
@dataclass(frozen=True)
class Conversation:
    head: EventNode | None = None

    def append(self, msg: Message) -> Conversation:
        depth = (self.head.depth + 1) if self.head else 0
        return Conversation(head=EventNode(message=msg, parent=self.head, depth=depth))

    def messages(self) -> Iterator[Message]:
        """Lazy walk from head to root, chronological order."""
        stack, node = [], self.head
        while node:
            stack.append(node.message)
            node = node.parent
        while stack:
            yield stack.pop()

    def __len__(self) -> int:
        return (self.head.depth + 1) if self.head else 0

    @staticmethod
    def start(*messages: Message) -> Conversation:
        log = Conversation()
        for msg in messages:
            log = log.append(msg)
        return log


# Assembles what the LLM sees — ephemeral, discarded after each call
def build_context(
    log: Conversation,
    system: str | None = None,
    injections: list[str] | None = None,
    window: int | None = None,
) -> list[Message]:
    context: list[Message] = []
    if system:
        context.append(Message("system", system))
    for text in (injections or []):
        context.append(Message("system", text))
    history = list(log.messages())
    if window:
        history = history[-window:]
    context.extend(history)
    return context