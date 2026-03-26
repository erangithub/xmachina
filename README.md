# XMachina

> _Deus ex machina_ — the god from the machine. Ancient playwrights lowered a god onto the stage when they'd lost control of the plot. Aristotle called it a cheat.

XMachina reframes agentic LLM systems as a **state machine** with an **environment**.

The environment is the source of all non-determinism — user input, LLM completions, tool results. The state machine is yours: deterministic, explicit, testable, expressed naturally in code.

This reframing dissolves problems that other frameworks solve with infrastructure:
checkpointing, branching, time travel, replay — all fall out naturally
from an immutable log and a clean separation of concerns.

This is **[event sourcing](https://martinfowler.com/eaaDev/EventSourcing.html)** applied to LLM conversations.

| Event Sourcing | XMachina                                                 |
| -------------- | -------------------------------------------------------- |
| Event Store    | Immutable node chain — O(1) branching                    |
| Event          | `Message`                                                |
| Projection     | `evolve(state, message)` — you write this                |
| Read Model     | `build_context(env.history())` — ephemeral               |
| Command        | `env.input()` / `env.llm_complete()` / `env.call_tool()` |

**You don't need the LLM to be deterministic. You need the log to be honest.**

---

## Requirements

- Python 3.10+

## Installation

```bash
pip install -e .
```

---

## Quick start

```python
from xmachina import Message, build_context
from xmachina.llms import OpenAILLM
from xmachina.environment import Environment

env = Environment(llm=OpenAILLM(), input_fn=input)

while True:
    # live: get user input, write to log         | replay: read from log
    request  = env.input()
    while True:
        context  = build_context(env.history())
        # live: call llm, record result to log   | replay: read from log
        response = env.llm_complete(context)
        if not response.tool_calls: break
        # live: run tools, record results to log | replay: read from log
        for tc in response.tool_calls: env.call_tool(tc)
```

Everything else — memory, checkpointing, branching, multi-agent, streaming — is a variation of this loop.

---

## Replay

The same environment, the same code, three modes.

### Full replay — no LLM calls

```python
from xmachina import Message, build_context
from xmachina.llms import EchoLLM
from xmachina.environment import Environment

# Record
env = Environment(llm=EchoLLM(), input_fn=input, continue_live=True)
env.add_message(Message("user", "hello"))
env.add_message(Message("assistant", "echo: hello"))

# Replay — reads from log, validates roles, no LLM called
env.rewind()
env.input()
response = env.llm_complete(build_context(env.history()))

assert response.content == "echo: hello"
```

### Replay then continue live

Pre-fill part of the conversation, replay it, then continue with a real LLM:

```python
env = Environment(llm=EchoLLM(), input_fn=input, continue_live=True)
env.add_message(Message("user", "hello"))

# Rewind — replays the user message, then calls LLM live for the response
env.rewind()
env.input()
response = env.llm_complete(build_context(env.history()))

assert response.content == "echo: hello"
```

### Pure live

```python
env = Environment(llm=EchoLLM(), input_fn=lambda: "hello")

env.input()
response = env.llm_complete(build_context(env.history()))

assert response.content == "echo: hello"
```

**The loop is identical in all three modes.** `rewind()` resets the environment. The code never changes.

---

## Forking

`env.fork()` creates a child environment that branches off the current position.
The fork shares the parent's history but has its own write head.
The parent log is never touched.

This is how summarization works without contaminating the conversation:

```python
from xmachina import build_context
from xmachina.llms import EchoLLM
from xmachina.environment import Environment


def summarize(env) -> str:
    """Summarize in a fork — never touches the parent log."""
    sub = env.fork()
    context = build_context(sub.history(), system="Summarize the conversation.")
    response = sub.llm_complete(context)
    return f"Summary: {response.content}"


def flow(env):
    env.add_message(Message("user", "long conversation here..."))

    # Fork to summarize — parent log unchanged
    summary = summarize(env)

    # Inject summary as context, not as a log entry
    context = build_context(env.history(), injections=[summary])
    response = env.llm_complete(context)
    return response


def main():
    llm = EchoLLM()

    # First run — live
    env = Environment(llm=llm, input_fn=input)
    flow(env)

    # Second run — exact replay, including the fork
    env.rewind()
    flow(env)
```

`flow()` is identical in both runs. The fork replays automatically on `rewind()`.
No mocks. No special test mode. No contamination.

Forking also enables parallelism:

```python
import asyncio

async def run_parallel(env):
    branch_a = env.fork()
    branch_b = env.fork()

    result_a, result_b = await asyncio.gather(
        run_branch(branch_a),
        run_branch(branch_b),
    )
```

Both branches share history. Both replay independently. Standard Python concurrency — no swarm orchestrator needed.

---

## Tools

```python
from xmachina.environment import Environment, Tool

tools = [
    Tool(name="get_weather", fn=get_weather, schema=weather_schema)
]

env = Environment(llm=MyLLM(), tools=tools, input_fn=input)

while True:
    request  = env.input()
    while True:
        context  = build_context(env.history())
        response = env.llm_complete(context)
        if not response.tool_calls: break
        for tc in response.tool_calls:
            env.call_tool(tc)   # executes tool, appends result to log
```

Schema generation belongs to the ecosystem — OpenAI SDK, Pydantic, FastMCP.
XMachina doesn't provide `@tool`. That would conflict with what you already have.

---

## LLM providers

```python
# OpenAI
from xmachina.llms import OpenAILLM
llm = OpenAILLM(api_key="...")

# Groq (free tier)
from xmachina.llms import GroqLLM
llm = GroqLLM(api_key="...")

# Ollama (local)
from xmachina.llms import OllamaLLM
llm = OllamaLLM(model="llama3.2")

# LM Studio (local)
from xmachina.llms import LMStudioLLM
llm = LMStudioLLM()

# Echo (testing)
from xmachina.llms import EchoLLM
llm = EchoLLM()
```

---

## Examples

See `examples/` for more:

### simple/
- `hello.py` — basic LLM call
- `streaming.py` — streaming responses
- `interactive.py` — interactive input loop
- `replay.py` — replay modes

### swarm/
- `swarm.py` — parallel forking with asyncio
- `branch_summarize.py` — forking for summarization

### tool_call/
- `tool_call_echo.py` — tool calling with EchoLLM
- `tool_call_gemini.py` — tool calling with Gemini (requires API key)
- `async_tool_call_gemini.py` — async tool calling (requires API key)

---

_The playwright who reaches for the crane because they ran out of ideas has a different relationship to their story than the one who picks it up deliberately. XMachina is for the second kind of engineer._
