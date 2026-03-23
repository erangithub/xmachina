from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Callable, TypeVar
from xmachina import MessageEvent, Delta, WriteHead, EventNode, ToolCall, Sequence, BRANCH_START, ControlEvent, CustomFunctionEvent, Event
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
    ):
        self.llm = llm
        self.tools = tools or []
        self.input_fn = input_fn

        self.origin_node = origin_node
        self.write_head = WriteHead(parent = origin_node)
        self.forks: dict[str, Environment] = {} # Environment forks from head node_ids

        self.rewind(continue_live)

    @property
    def is_replay(self) -> bool:
        return (self.readhead is not None)

    def _write(self, event: Event):
        if self.is_replay:
            raise RuntimeError("Cannot write in replay mode")
        self.write_head.append(event)
        self.child_index = 0

    def _read(self) -> Event | None:
        if self.readhead is None:
            return None
        self.lastread = next(self.readhead, None)
        self.child_index = 0
        if self.lastread is None:
            self.readhead = None
            return None
        return self.lastread.event

    def current_node(self) -> EventNode:
        return self.lastread if self.is_replay else self.write_head.parent
    
    def history(self) -> Sequence:
        return Sequence(after_node=None, to_node=self.current_node())

    def fork(self) -> Environment:
        if self.write_head.parent is None:
            raise RuntimeError("Cannot fork from empty log")
        forknode = self.current_node()
        if not forknode.id in self.forks:
            self.forks[forknode.id] = []
        child_envs = self.forks[forknode.id]
        if self.child_index < len(child_envs): # defacto, replay
            forked_env = child_envs[self.child_index]
            forked_env.rewind()
        else:
            forked_env = Environment(
                llm=self.llm,
                tools=self.tools,
                input_fn=self.input_fn,
                continue_live=self.continue_live,
                origin_node=self.write_head.fork(),
            )
            child_envs.append(forked_env)

        self.child_index += 1
        return forked_env
        

    def _invoke(self, to_message: Callable[[T], MessageEvent], fn: Callable[[], T], expected_role: str) -> MessageEvent:
        if self.is_replay:
            result = self._read()
            while result == BRANCH_START:
                result = self._read()
            if result is not None:
                if result.role != expected_role:
                    raise RuntimeError(f"Expected {expected_role}, got {result.role}")
                return result
            if not self.continue_live:
                raise RuntimeError("Replay exhausted")
        result = fn()
        self._write(to_message(result))
        return to_message(result)

    def llm_complete(self, messages: list[MessageEvent]) -> MessageEvent:
        return self._invoke(
            to_message=lambda m: m,
            fn=lambda: self.llm.complete(messages),
            expected_role="assistant",
        )

    def llm_stream(self, messages: list[MessageEvent]) -> Iterator[Delta]:
        if self.is_replay:
            result = self._read()
            if result is not None:
                if result.role != "assistant":
                    raise RuntimeError(f"Expected assistant, got {result.role}")
                if result.content:
                    for word in result.content.split():
                        yield Delta(word + " ")
                return
            if not self.continue_live:
                raise RuntimeError("Replay exhausted")
        deltas = list(self.llm.stream(messages))
        self._write(MessageEvent(role="assistant", content="".join(d.content for d in deltas)))
        yield from deltas

    def call_tool(self, tool_call: ToolCall) -> str:
        import json
        args = json.loads(tool_call.arguments)
        def fn():
            tool = next(t for t in self.tools if t.name == tool_call.name)
            return tool.fn(**args)
        result = self._invoke(
            to_message=lambda r: MessageEvent(role="tool", content=r, tool_call_id=tool_call.id),
            fn=fn,
            expected_role="tool",
        )
        return result.content

    def input(self) -> str:
        result = self._invoke(
            to_message=lambda t: MessageEvent(role="user", content=t),
            fn=self.input_fn,
            expected_role="user",
        )
        return result.content

    # Add a message to the log as if it came from the environment
    # this is useful for testing
    def add_message(self, role: str, content: str | None = None, **kwargs) -> str:
        msg = MessageEvent(role=role, content=content, **kwargs)
        result = self._invoke(
            to_message=lambda x: x,
            fn=lambda: msg,
            expected_role=msg.role,
        )
        return result.content

    # Add a message to the log as if it came from the environment
    # this is useful for testing
    def add_user_message(self, text: str) -> str:
        return self.add_message(MessageEvent(role="user", content=text))

    def nondet(self, fn: Callable[..., T]) -> Callable[..., T]:
        import json
        fn_name = fn.__name__

        def wrapper(*args, **kwargs):
            if self.is_replay:
                result = self._read()
                while result == BRANCH_START:
                    result = self._read()
                if result is not None:
                    if isinstance(result, CustomFunctionEvent):
                        return json.loads(result.content)
                    raise RuntimeError(f"Expected CustomFunctionEvent, got {type(result)}")
                if not self.continue_live:
                    raise RuntimeError("Replay exhausted")
            
            result = fn(*args, **kwargs)
            self._write(CustomFunctionEvent(fn_name=fn_name, content=json.dumps(result)))
            return result

        return wrapper
    
    def rewind(self, continue_live: bool | None = None):
        last_written_node = self.write_head.parent
        if last_written_node and last_written_node != self.origin_node :
            self.readhead = Sequence(after_node=self.origin_node, to_node=self.write_head.parent).iter_nodes()
        else:
            self.readhead = None

        if continue_live is not None:
            self.continue_live = continue_live
        self.child_index = 0
        self.lastread = self.origin_node