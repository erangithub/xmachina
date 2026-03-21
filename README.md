# XMachina

> _Deus ex machina_ — the god from the machine. Ancient playwrights lowered a god onto the stage when they'd lost control of the plot. Aristotle called it a cheat.

XMachina reframes agentic LLM systems as a **state machine** with an **environment**.

The environment is the source of all non-determinism — user input, LLM completions, tool results. The state machine is yours: deterministic, explicit, testable, expressed naturally in code.

This reframing dissolves problems that other frameworks solve with infrastructure:
checkpointing, branching, time travel, replay, runtime — all fall out naturally
from an immutable log and a clean separation of concerns.

```
input, llm, tools  →  Message  →  evolve(state, msg)  →  build_context  →  LLM
```

Three separations follow naturally:

- **Log vs state** — the log is immutable and append-only; state is derived from it by a fold. Many agents don't have any state beyond the log.
- **State vs view** — `build_context` assembles what the LLM sees, ephemerally, from state
- **Live vs replay** — swap the environment to replay any session exactly, no mocks needed

This is **event sourcing** applied to LLM conversations.

| Event Sourcing | XMachina                                               |
| -------------- | ------------------------------------------------------ |
| Event Store    | `EventLog` (immutable tree, O(1) branching)            |
| Event          | `Message`                                              |
| Projection     | `evolve(state, message)` — you write this              |
| Read Model     | `build_context` — ephemeral, discarded after each call |
| Command        | `env.input()` / `env.llm_complete()` / `env.call_tool()` |

The canonical loop:

```python
env = live(llm=MyLLM(), input_fn=input)
# or: env = replay(saved_log, continue_live=True)

while True:
    request  = env.input()
    context  = build_context(env.log)
    response = env.llm_complete(context)
```

Everything else — memory, checkpointing, branching, multi-agent, streaming — is a variation of this loop.

**You don't need the LLM to be deterministic. You need the log to be honest.**

---

## Requirements

- Python 3.10+

## Installation

```bash
pip install -e .
```

This installs xmachina with its only dependency: `openai>=1.0`.

## Quick start

```python
from xmachina import Message, EventLog, build_context
from xmachina.llms import OpenAILLM
from xmachina.environment import live

env = live(llm=OpenAILLM(), input_fn=input)

while True:
    user_input = env.input()
    if user_input.lower() in ("quit", "exit"):
        break
    context = build_context(env.log)
    response = env.llm_complete(context)
    print(response.content)
```

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

## Replay

Record a session, replay it exactly:

```python
from xmachina import Message, EventLog, build_context
from xmachina.llms import EchoLLM
from xmachina.environment import replay

log = EventLog()
log = log.append(Message("user", "hello"))
log = log.append(Message("assistant", "echo: hello"))

env = replay(log, llm=EchoLLM())
env.input()  # replays user message
response = env.llm_complete(build_context(env.log))
```

Or replay then continue live:

```python
env = replay(log, llm=MyLLM(), input_fn=input, continue_live=True)
```

## Examples

See `examples/` for more:

- `examples/hello.py` — basic LLM call
- `examples/echo/` — testing with EchoLLM

---

_The playwright who reaches for the crane because they ran out of ideas has a different relationship to their story than the one who picks it up deliberately. XMachina is for the second kind of engineer._
