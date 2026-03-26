Here's the updated README:

---

# XMachina

> _Deus ex machina_ — the god from the machine. Ancient playwrights lowered a god onto the stage when they'd lost control of the plot. Aristotle called it a cheat.

XMachina reframes agentic LLM systems as a **state machine** with an **environment**.

The environment is the source of all non-determinism — user input, LLM completions, tool results. The state machine is yours: deterministic, explicit, testable, expressed naturally in code.

This reframing dissolves problems that other frameworks solve with infrastructure:
checkpointing, branching, time travel, replay — all fall out naturally
from an immutable log and a clean separation of concerns.

This is **[event sourcing](https://martinfowler.com/eaaDev/EventSourcing.html)** applied to LLM conversations.

| Event Sourcing | XMachina                                                                  |
| -------------- | ------------------------------------------------------------------------- |
| Event Store    | Immutable node chain — O(1) branching                                     |
| Event          | `MessageEvent` / `CallEvent` / `ControlEvent`                             |
| Projection     | `build_context(env.history())` — ephemeral                                |
| Command        | `env.llm_complete()` / `env.input()` / `env.call_tool()` / `env.nondet()` |

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
from xmachina import build_context
from xmachina.llms import OpenAILLM
from xmachina.environment import Environment

env = Environment()
env.register_llm_fn(OpenAILLM().complete)
env.register_input_fn(input)

while True:
    # live: get user input, write to log         | replay: read from log
    env.input()
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

The most underrated problem in LLM engineering: you can't reproduce what happened.

XMachina solves this structurally. Every environment call writes to an immutable log. The environment itself is the persistent thing — `rewind()` just moves the read head.

**Three modes, one environment:**

```python
from xmachina import build_context
from xmachina.llms import EchoLLM
from xmachina.environment import Environment

env = Environment(continue_live=True)
env.register_llm_fn(EchoLLM().complete)
env.register_input_fn(input)

env.add_message("user", "hello")
env.add_message("assistant", "echo: hello")

# Full replay — no LLM calls, reads entirely from log
env.rewind()
env.input()
response = env.llm_complete(build_context(env.history()))
assert response.content == "echo: hello"

# Replay then continue live
env.add_message("user", "hello")
env.rewind(continue_live=True)
env.input()                                          # replay
response = env.llm_complete(build_context(env.history()))  # live

# Pure live
env = Environment(continue_live=True)
env.register_llm_fn(EchoLLM().complete)
env.register_input_fn(lambda: "hello")
env.input()
response = env.llm_complete(build_context(env.history()))
```

**The loop is identical in all three modes.** `rewind()` resets the environment. The code never changes.

---

## Forking

`env.fork()` creates a child environment that branches off the current position. The fork shares the parent's history but has its own write head. The parent log is never touched.

This is how summarization works without contaminating the conversation:

```python
from xmachina import build_context
from xmachina.llms import EchoLLM
from xmachina.environment import Environment


def summarize(env) -> str:
    sub = env.fork()
    context = build_context(sub.history(), system="Summarize the conversation.")
    response = sub.llm_complete(context)
    return response.content


def flow(env):
    env.add_message("user", "long conversation here...")

    # Fork to summarize — parent log unchanged
    summary = summarize(env)

    # Inject summary as context, not as a log entry
    context = build_context(env.history(), injections=[summary])
    return env.llm_complete(context)


llm = EchoLLM()
env = Environment(continue_live=True)
env.register_llm_fn(llm.complete)

flow(env)       # live

env.rewind()
flow(env)       # exact replay, including the fork
```

`flow()` is identical in both runs. The fork replays automatically on `rewind()`. No mocks. No special test mode. No contamination.

Forking also enables parallelism:

```python
import asyncio

env = Environment(continue_live=True)
env.register_llm_afn(llm.acomplete)

# Create forks sequentially to lock in log order, then run in parallel
branch_a = env.fork()
branch_b = env.fork()

result_a, result_b = await asyncio.gather(
    run_branch(branch_a),
    run_branch(branch_b),
)
```

Both branches share history. Both replay independently. Standard Python concurrency — no swarm orchestrator needed.

---

## Time Travel

The immutable log isn't just for replay — it's a complete history you can navigate.
delorean.py shows a chatbot where the user can move backwards and forwards through their conversation history, then branch off in a new direction. No special infrastructure. Just the log, a read head, and two predicates.

```python
if user_input == "b":
    # walk back to the previous user message
    target = env.prev_node.find(lambda n: n.is_message("user"))
    if target is None:
        print('Nothing to go back to')
        continue
    env.rewind()
    env.replay_until(lambda n, d=target.depth: n.depth >= d)

if user_input == "n":
    # walk forward to the next user message
    current_depth = env.current_depth
    env.replay_until(lambda n, d=current_depth: n.is_message("user") and n.depth > d)
```

Going back rewinds the log and replays up to a previous point. Going forward advances
the read head to the next recorded user message. When the user types something new at
any point — whether after going all the way back, or stopping partway through — the
write head syncs to the current position and continues from there. Previous nodes
remain in the log, unreachable but intact. A new branch grows without any explicit
fork call.

This is not a feature XMachina implements. It falls out of the model.

---

## Tools

```python
from xmachina import build_context
from xmachina.environment import Environment, Tool

tools = [
    Tool(name="get_weather", fn=get_weather, schema=weather_schema)
]

env = Environment()
env.register_llm_fn(llm.complete)
env.register_input_fn(input)
env.register_tool_fns(tools)

while True:
    env.input()
    while True:
        context  = build_context(env.history())
        response = env.llm_complete(context, tools=[t.schema for t in tools])
        if not response.tool_calls: break
        for tc in response.tool_calls:
            env.call_tool(tc)   # executes tool, appends result to log
```

Schema generation belongs to the ecosystem — OpenAI SDK, Pydantic, FastMCP. XMachina doesn't provide `@tool`. That would conflict with what you already have.

---

## Non-deterministic functions

Any function that reaches outside the process — API calls, random numbers, time, file reads — can be made replayable with `nondet`:

```python
env = Environment(continue_live=True)
env.register_llm_fn(llm.complete)

det_get_price = env.nondet(get_price)  # lift into logged, replayable context

price = det_get_price("AAPL")  # live: calls API, writes to log
                                # replay: reads from log
```

For functions called repeatedly, lift once at the top of your flow:

```python
def my_flow(env):
    det_get_price = env.nondet(get_price)
    det_time      = env.nondet(time.time)

    price = det_get_price("AAPL")
    ts    = det_time()
    ...
```

For persistent registration across forks, use `register_nondet`:

```python
env.register_nondet(get_price)
env.get_price("AAPL")  # available on env and all its forks
```

---

## LLM providers

```python
from xmachina.llms import OpenAILLM    # OpenAI
from xmachina.llms import GroqLLM      # Groq (free tier)
from xmachina.llms import OllamaLLM    # Ollama (local)
from xmachina.llms import LMStudioLLM  # LM Studio (local)
from xmachina.llms import EchoLLM      # Testing

env = Environment(continue_live=True)
env.register_llm_fn(OpenAILLM(api_key="...").complete)

# Async
env.register_llm_afn(OpenAILLM(api_key="...").acomplete)

# Streaming
env.register_llm_stream_fn(OpenAILLM(api_key="...").stream)
```

---

## Debugging

```python
env.print_tree()
```

Prints the full event log as a tree, including forks and their branches. Useful for understanding what actually happened during a run.

---

## Examples

See `examples/` for more:

### simple/

- `hello.py` — basic LLM call
- `streaming.py` — streaming responses
- `interactive.py` — interactive input loop
- `replay.py` — replay modes

### nondet/

- `example.py` — non-deterministic function replay

### swarm/

- `swarm.py` — parallel forking with asyncio
- `branch_summarize.py` — forking for summarization

### tool_call/

- `tool_call_echo.py` — tool calling with EchoLLM
- `tool_call_gemini.py` — tool calling with Gemini (requires API key)
- `async_tool_call_gemini.py` — async tool calling (requires API key)

### time_travel/

- `delorean.py` — navigate backwards and forwards through conversation history

---

_The playwright who reaches for the crane because they ran out of ideas has a different relationship to their story than the one who picks it up deliberately. XMachina is for the second kind of engineer._
