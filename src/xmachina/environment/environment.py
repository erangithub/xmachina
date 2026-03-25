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
        tools: list[Tool] | None = None,
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
            if hasattr(llm, 'stream'):
                self.register_llm_stream_fn(llm.stream)
        if input_fn is not None:
            self.register_input_fn(input_fn)
        if tools:
            self.register_tool_fns(tools)

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
                registered_fns=self.registered_fns,  # inherited
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

    def add_message(self, role: str, content: str | None = None, **kwargs) -> str:
        if not isinstance(role, str):
            raise RuntimeError(f"Wrote type {type(role)} passed for 'role' argument in add_message, it must be a sstring.")
        msg = Message(role=role, content=content, **kwargs)
        result = self._message_event(fn=lambda: msg)
        return result.content

    def add_user_message(self, text: str) -> str:
        return self.add_message(role="user", content=text)

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
        self._register_method(
            name,
            lambda obj, *args, **kwargs: obj._message_event(
                fn=lambda: fn(*args, **kwargs),
            ),
        )

    def register_llm_stream_fn(self, fn: Callable, name: str = "llm_stream"):
        def method(obj, messages):
            content_parts = []
            if not obj.is_replay:
                for delta in fn(messages):
                    content_parts.append(delta.content)
                    yield delta
            # log the message
            msg = obj._message_event(
                fn=lambda: Message(role="assistant", content="".join(content_parts))
            )
            if obj.is_replay:
                for word in (msg.content or "").split():
                    yield Delta(content=word + " ")
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
            self.readhead = Sequence(
                after_node=self.origin_node,
                to_node=self.write_head.parent,
            ).iter_nodes()
        else:
            self.readhead = None

        if continue_live is not None:
            self.continue_live = continue_live
        self.child_index = 0
        self.lastread = self.origin_node