from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Callable, TypeVar, Generic
from xmachina import Message, Delta, EventLog
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
        log: EventLog | None = None,
        readhead: Iterator[Message] | None = None,
        llm: LLM | None = None,
        tools: list[Tool] | None = None,
        input_fn: Callable[[], str] | None = None,
        continue_live: bool = False,
    ):
        self.log = log or EventLog()
        self.readhead = readhead
        self.llm = llm
        self.tools = tools or []
        self.input_fn = input_fn
        self.continue_live = continue_live

    @property
    def is_replay(self) -> bool:
        return self.readhead is not None

    def _write(self, msg: Message):
        self.log = self.log.append(msg)

    def _read(self) -> Message | None:
        if self.readhead is None:
            return None
        result = next(self.readhead, None)
        if result is None and self.continue_live:
            self.readhead = None
        return result

    def _invoke(self, to_message: Callable[[T], Message], fn: Callable[[], T]) -> T:
        if self.is_replay:
            result = self._read()
            if result is not None:
                return to_message(result)
            if not self.continue_live:
                raise RuntimeError("Replay exhausted")
        result = fn()
        self._write(to_message(result))
        return result

    def llm_complete(self, messages: list[Message]) -> Message:
        return self._invoke(
            to_message=lambda m: m,
            fn=lambda: self.llm.complete(messages),
        )

    def llm_stream(self, messages: list[Message]) -> Iterator[Delta]:
        if self.is_replay:
            result = self._read()
            if result is not None:
                if result.content:
                    for word in result.content.split():
                        yield Delta(word + " ")
                return
            if not self.continue_live:
                raise RuntimeError("Replay exhausted")
        deltas = list(self.llm.stream(messages))
        self._write(Message(role="assistant", content="".join(d.content for d in deltas)))
        yield from deltas

    def call_tool(self, name: str, tool_call_id: str, arguments: dict) -> str:
        def fn():
            tool = next(t for t in self.tools if t.name == name)
            return tool.fn(**arguments)
        return self._invoke(
            to_message=lambda r: Message(role="tool", content=r, tool_call_id=tool_call_id),
            fn=fn,
        )

    def input(self) -> str:
        return self._invoke(
            to_message=lambda t: Message(role="user", content=t),
            fn=self.input_fn,
        )


def replay(log: EventLog, llm: LLM | None = None, tools: list[Tool] | None = None, input_fn: Callable[[], str] | None = None, continue_live: bool = False) -> Environment:
    return Environment(log=log, readhead=iter(log.messages()), llm=llm, tools=tools, input_fn=input_fn, continue_live=continue_live)


def live(llm: LLM, tools: list[Tool] | None = None, input_fn: Callable[[], str] | None = None) -> Environment:
    return Environment(log=EventLog(), readhead=None, llm=llm, tools=tools, input_fn=input_fn)
