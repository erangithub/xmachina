from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterator
import uuid


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class Message:
    role: str
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None


@dataclass(frozen=True)
class Delta:
    content: str


DUMMY = Message("", "")


@dataclass(frozen=True)
class EventNode:
    id: str
    message: Message
    parent: EventNode | None
    depth: int = 0


class Sequence:
    after_node: EventNode | None # Sequence starts after this node, or None for start from beginning
    to_node: EventNode           # Sequence ends at this node (included)

    def __init__(self, after_node: EventNode | None, to_node: EventNode):
        self.after_node = after_node
        self.to_node = to_node
        
    # Returns iterator over all messages
    def iter_messages(self) -> Iterator[Message]:
        stack = []
        node = self.to_node
        while node and (node != self.after_node):
            if node.message != DUMMY:
                stack.append(node.message)
            node = node.parent
        while stack:
            yield stack.pop()

    # Returns iterator over all nodes 
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

    def append(self, msg: Message):
        node_id = str(uuid.uuid4())
        depth = (self.parent.depth + 1) if self.parent else 0
        self.parent = EventNode(id=node_id, message=msg, parent=self.parent, depth=depth)

    def fork(self) -> EventNode:
        node_id = str(uuid.uuid4())
        depth = (self.parent.depth + 1) if self.parent else 0
        return EventNode(id=node_id, message=DUMMY, parent=self.parent, depth=depth)

    @staticmethod
    def start(*messages: Message) -> WriteHead:
        write_head = WriteHead()
        for msg in messages:
            write_head.append(msg)
        return write_head


def build_context(
    sequence: Sequence,
    system: str | None = None,
    injections: list[str] | None = None
) -> list[Message]:
    context: list[Message] = []
    if system:
        context.append(Message("system", system))
    for text in (injections or []):
        context.append(Message("system", text))
    history = list(sequence.iter_messages())
    context.extend(history)
    return context
