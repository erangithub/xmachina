from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, Any
import json
import uuid


@dataclass(frozen=True)
class Delta:
    content: str


@dataclass(frozen=True)
class Message:
    role: str
    content: str | None = None
    tool_calls: tuple = ()
    tool_call_id: str | None = None


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


class ControlKind(Enum):
    branch_start = "branch_start"
    branch_cancelled = "branch_cancelled"


@dataclass
class ControlEvent:
    control: ControlKind


@dataclass
class MessageEvent:
    message: Message
    timestamp: float | None = None
    duration_ms: float | None = None


@dataclass
class CallEvent:
    fn_name: str
    result: str
    timestamp: float | None = None
    duration_ms: float | None = None
    args: str | None = None


Event = MessageEvent | CallEvent | ControlEvent


@dataclass(frozen=True)
class EventNode:
    id: str
    event: Event
    parent: EventNode | None
    depth: int = 0


class Sequence:
    after_node: EventNode | None
    to_node: EventNode

    def __init__(self, after_node: EventNode | None, to_node: EventNode):
        self.after_node = after_node
        self.to_node = to_node

    def iter_messages(self) -> Iterator[Message]:
        for node in self.iter_nodes():
            if isinstance(node.event, MessageEvent):
                yield node.event.message

    def iter_nodes(self) -> Iterator[EventNode]:
        stack = []
        node = self.to_node
        while node and (node != self.after_node):
            stack.append(node)
            node = node.parent
        while stack:
            yield stack.pop()


class WriteHead:
    parent: EventNode | None

    def __init__(self, parent: EventNode | None = None):
        self.parent = parent

    def append(self, event: Event):
        node_id = str(uuid.uuid4())
        depth = (self.parent.depth + 1) if self.parent else 0
        self.parent = EventNode(id=node_id, event=event, parent=self.parent, depth=depth)

    def fork(self) -> EventNode:
        node_id = str(uuid.uuid4())
        depth = (self.parent.depth + 1) if self.parent else 0
        return EventNode(
            id=node_id,
            event=ControlEvent(control=ControlKind.branch_start),
            parent=self.parent,
            depth=depth,
        )

    @staticmethod
    def start() -> WriteHead:
        return WriteHead()


def build_context(
    sequence: Sequence,
    system: str | None = None,
    injections: list[str] | None = None,
) -> list[Message]:
    context: list[Message] = []
    if system:
        context.append(Message(role="system", content=system))
    for text in (injections or []):
        context.append(Message(role="system", content=text))
    for msg in sequence.iter_messages():
        context.append(msg)
    return context