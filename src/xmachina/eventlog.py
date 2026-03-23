from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Iterator
import uuid


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class MessageEvent:
    role: str
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None


@dataclass(frozen=True)
class Delta:
    content: str


@dataclass
class CustomFunctionEvent:
    fn_name: str
    content: str


class ControlKind(Enum):
    branch_start = "branch_start"
    branch_cancelled = "branch_cancelled"


@dataclass
class ControlEvent:
    control: ControlKind


Event = MessageEvent | CustomFunctionEvent | ControlEvent

BRANCH_START = ControlEvent(control=ControlKind.branch_start)


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

    def iter_messages(self) -> Iterator[MessageEvent]:
        stack = []
        node = self.to_node
        while node and (node != self.after_node):
            if isinstance(node.event, MessageEvent):
                stack.append(node.event)
            node = node.parent
        while stack:
            yield stack.pop()

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
        return EventNode(id=node_id, event=BRANCH_START, parent=self.parent, depth=depth)

    @staticmethod
    def start(*messages: MessageEvent) -> WriteHead:
        write_head = WriteHead()
        for msg in messages:
            write_head.append(msg)
        return write_head


def build_context(
    sequence: Sequence,
    system: str | None = None,
    injections: list[str] | None = None
) -> list[MessageEvent]:
    context: list[MessageEvent] = []
    if system:
        context.append(MessageEvent("system", system))
    for text in (injections or []):
        context.append(MessageEvent("system", text))
    history = list(sequence.iter_messages())
    context.extend(history)
    return context
