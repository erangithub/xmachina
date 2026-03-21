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
    depth: 0


class EventStore:
    def __init__(self):
        self.nodes: dict[str, EventNode] = {}
        self.children: dict[str, list[str]] = {}

    def add(self, node: EventNode):
        self.nodes[node.id] = node
        parent = node.parent
        if parent is not None:
            if parent.id not in self.children:
                self.children[parent.id] = []
            self.children[parent.id].append(node.id)

    def get_children(self, node_id: str) -> list[EventNode]:
        return [self.nodes[cid] for cid in self.children.get(node_id, [])]

    def left_most_descendant(self, node: EventNode) -> str:
        while True:
            children = self.get_children(node.id)
            if not children:
                return node
            node = children[0]


@dataclass
class EventLog:
    head: EventNode | None = None
    store: EventStore = field(default_factory=EventStore)

    def append(self, msg: Message) -> EventLog:
        node_id = str(uuid.uuid4())
        node = EventNode(id=node_id, message=msg, parent=self.head, depth=len(self)+1)
        self.store.add(node)
        return EventLog(head=node, store=self.store)

    # Returns iterator over all messages
    def messages(self) -> Iterator[Message]:
        stack = []
        node = self.head
        while node:
            if node.message != DUMMY:
                stack.append(node.message)
            node = node.parent
        while stack:
            yield stack.pop()

    # Returns iterator over all nodes start at 'start_id' (not included)
    # and leading upto the head (included)
    def nodes(self, start_id: str | None = None) -> Iterator[EventNode]:
        stack = []
        node = self.head
        while node and node.id != start_id:
            if node.message != DUMMY:
                stack.append(node)
            node = node.parent
        while stack:
            yield stack.pop()

    def read_from_here(self) -> Iterator[EventNode]:
        deepest = self.store.left_most_descendant(self.head)
        deepLog = EventLog(deepest, self.store)
        return deepLog.nodes(self.head.id)

    def __len__(self) -> int:
        return self.head.depth if self.head else 0

    @staticmethod
    def start(*messages: Message) -> EventLog:
        log = EventLog()
        for msg in messages:
            log = log.append(msg)
        return log


def build_context(
    log: EventLog,
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
