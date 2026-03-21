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


class EventStore:
    def __init__(self):
        self.nodes: dict[str, EventNode] = {}
        self.children: dict[str, list[str]] = {}
        self.parents: dict[str, str] = {}

    def add(self, node: EventNode, parent_id: str | None = None):
        self.nodes[node.id] = node
        if parent_id is not None:
            if parent_id not in self.children:
                self.children[parent_id] = []
            self.children[parent_id].append(node.id)
            self.parents[node.id] = parent_id

    def get_children(self, node_id: str) -> list[EventNode]:
        
        return [self.nodes[cid] for cid in self.children.get(node_id, [])]

    def get_parent(self, node_id: str) -> EventNode | None:
        parent_id = self.parents.get(node_id)
        if parent_id:
            return self.nodes.get(parent_id)
        return None

    def left_most_descendant(self, node_id: str) -> str:
        while True:
            children = self.get_children(node_id)
            if not children:
                return node_id
            node_id = children[0].id


@dataclass
class EventLog:
    head_id: str | None = None
    store: EventStore = field(default_factory=EventStore)

    def append(self, msg: Message) -> EventLog:
        node_id = str(uuid.uuid4())
        node = EventNode(id=node_id, message=msg)
        self.store.add(node, self.head_id)
        return EventLog(head_id=node_id, store=self.store)

    def messages(self, start_id: str | None = None) -> Iterator[Message]:
        stack = []
        node_id = self.head_id
        while node_id and node_id != start_id:
            node = self.store.nodes.get(node_id)
            if node and node.message != DUMMY:
                stack.append(node.message)
            node_id = self.store.parents.get(node_id)
        while stack:
            yield stack.pop()

    def read_from_here(self) -> Iterator[Message]:
        deepest = self.store.left_most_descendant(self.head_id)
        deepLog = EventLog(deepest, self.store)
        return deepLog.messages(self.head_id)


    def __len__(self) -> int:
        if not self.head_id:
            return 0
        count = 0
        node_id = self.head_id
        while node_id:
            count += 1
            parent = self.store.get_parent(node_id)
            node_id = parent.id if parent else None
        return count

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
