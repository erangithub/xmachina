from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, TypeVar
from types import MethodType
import json
import functools

from xmachina.eventlog import (
    WriteHead, EventNode, ToolCall, ControlEvent, Message, Delta, Sequence,
    MessageEvent, CallEvent, Event
)
from xmachina.llms.base import LLM


T = TypeVar("T")


@dataclass
class Tool:
    name: str
    fn: Callable[..., str]
    schema: dict


class Environment:
    def __init__(
        self,
        llm: LLM | None = None,
        input_fn: Callable[[], str] | None = None,
        continue_live: bool = False,
        origin_node: EventNode | None = None,
        registered_fns: dict[str, Callable] | None = None,
    ):
        self.registered_fns: dict[str, Callable] = {}
        for name, fn in (registered_fns or {}).items():
            setattr(self, name, MethodType(fn, self))
            self.registered_fns[name] = fn

        self.origin_node = origin_node
        self.write_head = WriteHead(parent=origin_node)
        self.forks: dict[str, list[Environment]] = {}

        if llm is not None:
            self.register_llm_fn(llm.complete)
            if hasattr(llm, 'acomplete'):
                self.register_llm_afn(llm.acomplete)
            if hasattr(llm, 'stream'):
                self.register_llm_stream_fn(llm.stream)
        if input_fn is not None:
            self.register_input_fn(input_fn)

        self.rewind(continue_live)

    @property
    def is_replay(self) -> bool:
        return self.readhead is not None

    def _write(self, event: Event):
        if self.is_replay:
            raise RuntimeError("Cannot write in replay mode")
        self.write_head.append(event)
        self.child_index = 0

    def _read(self) -> MessageEvent | CallEvent | None:
        if self.readhead is None:
            return None
        self.lastread = next(self.readhead, None)
        self.child_index = 0
        if self.lastread is None:
            self.readhead = None
            return None
        while isinstance(self.lastread.event, ControlEvent):
            self.lastread = next(self.readhead, None)
            if self.lastread is None:
                self.readhead = None
                return None
        return self.lastread.event

    @property
    def current_node(self) -> EventNode:
        return (self.lastread if self.is_replay else self.write_head.parent) or self.origin_node

    def history(self) -> Sequence:
        return Sequence(after_node=None, to_node=self.current_node)

    def fork_history(self) -> Sequence:
        return Sequence(after_node=self.origin_node, to_node=self.current_node)
    
    def fork(self) -> Environment:
        forknode = self.current_node
        if forknode is None:
            raise RuntimeError("Cannot fork from empty log")
        if forknode.id not in self.forks:
            self.forks[forknode.id] = []
        child_envs = self.forks[forknode.id]
        if self.child_index < len(child_envs):
            forked_env = child_envs[self.child_index]
            forked_env.rewind()
        else:
            forked_env = Environment(
                continue_live=self.continue_live,
                origin_node=self.write_head.fork(),
                registered_fns=self.registered_fns,
            )
            child_envs.append(forked_env)

        self.child_index += 1
        return forked_env

    # --- core invoke primitives ---

    def _message_event(self, fn: Callable[[], Message]) -> Message:
        """Invoke a function that produces a Message. Writes a MessageEvent."""
        if self.is_replay:
            event = self._read()
            if event is not None:
                if not isinstance(event, MessageEvent):
                    raise RuntimeError(f"Expected MessageEvent, got {type(event)}")
                return event.message
            if not self.continue_live:
                raise RuntimeError("Replay exhausted")
        result = fn()
        if not isinstance(result, Message):
            raise RuntimeError(f"Expected Message, got {type(result)}")
        self._write(MessageEvent(message=result))
        return result

    async def _amessage_event(self, fn):
        """Invoke an async function that produces a Message. Writes a MessageEvent."""
        if self.is_replay:
            event = self._read()
            if event is not None:
                if not isinstance(event, MessageEvent):
                    raise RuntimeError(f"Expected MessageEvent, got {type(event)}")
                return event.message
            if not self.continue_live:
                raise RuntimeError("Replay exhausted")
        result = await fn()
        if not isinstance(result, Message):
            raise RuntimeError(f"Expected Message, got {type(result)}")
        self._write(MessageEvent(message=result))
        return result

    def _call_event(self, fn_name: str, fn: Callable[[], T]) -> T:
        """Invoke a non-deterministic function. Writes a CallEvent."""
        if self.is_replay:
            event = self._read()
            if event is not None:
                if event.fn_name != fn_name:
                    raise RuntimeError(f"Expected {fn_name}, got {event.fn_name}")
                if not isinstance(event, CallEvent):
                    raise RuntimeError(f"Expected CallEvent, got {type(event)}")
                return json.loads(event.result)
            if not self.continue_live:
                raise RuntimeError("Replay exhausted")
        result = fn()
        self._write(CallEvent(fn_name=fn_name, result=json.dumps(result)))
        return result

    # --- built-in env methods ---

    def call_tool(self, tool_call: ToolCall) -> str:
        tool_fn_name = f"tool.{tool_call.name}"
        tool_method = getattr(self, tool_fn_name, None)
        if tool_method is None:
            raise RuntimeError(f"Tool {tool_call.name} not registered.")
        args = json.loads(tool_call.arguments)
        msg = tool_method(**args)
        return msg.content

    def add_message(self, role: str | Message, content: str | None = None, **kwargs) -> str:
        if isinstance(role, Message):
            msg = role
        elif not isinstance(role, str):
            raise RuntimeError(f"Wrote type {type(role)} passed for 'role' argument in add_message, it must be a string.")
        else:
            msg = Message(role=role, content=content, **kwargs)
        return self._message_event(fn=lambda: msg)

    def add_user_message(self, text: str) -> str:
        return self.add_message(role="user", content=text).content

    # --- nondet / register ---

    def _wrap_message(self, fn: Callable[..., T], role: str) -> Callable[..., Message]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> Message:
            result = fn(*args, **kwargs)
            return Message(role=role, content=result)
        return wrapper

    def nondet(self, fn: Callable[..., T]) -> Callable[..., T]:
        fn_name = fn.__name__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return self._call_event(fn_name=fn_name, fn=lambda: fn(*args, **kwargs))
        return wrapper

    def _register_method(self, name: str, wrapper: Callable):
        self.registered_fns[name] = wrapper
        setattr(self, name, MethodType(wrapper, self))

    def register_llm_fn(self, fn: Callable[..., Message], name: str = "llm_complete"):
        def call_llm(obj, *args, **kwargs):
            return obj._message_event(
                fn=lambda: fn(*args, **kwargs),
            )
        
        self._register_method(name, call_llm)

    def register_llm_afn(self, fn, name: str = "llm_acomplete"):
        async def call_llm(obj, *args, **kwargs):
            return await obj._amessage_event(
                fn=lambda: fn(*args, **kwargs),
            )
        
        self._register_method(name, call_llm)

    def register_llm_stream_fn(self, fn: Callable, name: str = "llm_stream"):
        def method(obj, messages, **kwargs):
            content_parts = []
            for delta in fn(messages, **kwargs):
                content_parts.append(delta.content)
                yield delta
            
            obj._message_event(
                fn=lambda: Message(role="assistant", content="".join(content_parts))
            )
        
        self._register_method(name, method)

    def register_llm_astream_fn(self, fn, name: str = "llm_stream"):
        async def method(obj, messages, **kwargs):
            content_parts = []
            async for delta in fn(messages, **kwargs):
                content_parts.append(delta.content)
                yield delta
            
            await obj._amessage_event(
                fn=lambda: Message(role="assistant", content="".join(content_parts))
            )
        
        self._register_method(name, method)

    def register_nondet(self, fn: Callable[..., T], name: str | None = None):
        name = name or fn.__name__
        self._register_method(
            name,
            lambda obj, *args, **kwargs: obj._call_event(
                fn_name=name,
                fn=lambda: fn(*args, **kwargs),
            ),
        )

    def register_input_fn(self, fn: Callable[[], str], name: str = "input"):
        wrapped = self._wrap_message(fn, "user")
        self._register_method(
            name,
            lambda obj: obj._message_event(fn=wrapped).content,
        )

    def register_tool_fns(self, tools: list[Tool]):
        for tool in tools:
            fn_name = f"tool.{tool.name}"
            def make_method(f, n):
                return lambda obj, **kwargs: obj._message_event(
                    fn=lambda: Message(role="tool", content=f(**kwargs)),
                )
            self._register_method(fn_name, make_method(tool.fn, fn_name))

    def rewind(self, continue_live: bool | None = None):
        last_written_node = self.write_head.parent
        if last_written_node and last_written_node != self.origin_node:
            self.readhead = self.fork_history().iter_nodes()
        else:
            self.readhead = None

        if continue_live is not None:
            self.continue_live = continue_live
        self.child_index = 0
        self.lastread = self.origin_node

    def print_tree(self):
        """Print the event log as a tree."""
        history = self.history()
        self._print_tree_recursive(list(history.iter_nodes()), "", True)

    def _print_tree_recursive(self, nodes: list, prefix: str, is_last: bool):
        for i, node in enumerate(nodes):
            node_is_last = (i == len(nodes) - 1)
            connector = "└── " if node_is_last else "├── "
            new_prefix = prefix + ("    " if node_is_last else "│   ")
            
            if isinstance(node.event, MessageEvent):
                msg = node.event.message
                content = msg.content[:50] + "..." if msg.content and len(msg.content) > 50 else msg.content
                print(f"{prefix}{connector}[{msg.role}] {content}")
            
            child_forks = self.forks.get(node.id, [])
            for j, fork_env in enumerate(child_forks):
                fork_is_last = (j == len(child_forks) - 1)
                fork_connector = "└── " if fork_is_last else "├── "
                print(f"{new_prefix}{fork_connector}Fork {j+1}")
                fork_prefix = new_prefix + ("    " if fork_is_last else "│   ")
                fork_nodes = list(fork_env.fork_history().iter_nodes())
                fork_env._print_tree_recursive(fork_nodes, fork_prefix, fork_is_last)
