# XMachina

> _Deus ex machina_ — the god from the machine. Ancient playwrights lowered a god onto the stage when they'd lost control of the plot. The god resolved everything. The audience was impressed. Aristotle called it a cheat.
>
> In some ways, we are doing it again.

---

In every program you have ever debugged, one thing was settled: **the program holds the instruction pointer.** The CPU executes what you wrote. The user provides input. The program decides what to do with it. Even the most complex systems — operating systems, compilers, distributed databases — were ultimately state machines. The programmer was always, at every transition, in control.

Now there is something in the middle that is not deterministic, can reason, can decide what to do next, can take actions in the world, and produces different outputs on identical inputs.

That changes the architectural question.

There are two engineering temperaments emerging around LLMs. Some engineers are comfortable letting the model drive — and build accordingly. Others want the model to advise and the program to decide. Neither is wrong. XMachina is for the second temperament.

Distilled to its essence, this is a question of control flow: **does the LLM hold the instruction pointer, or does the program?**

One camp says yes. Give it the wheel. Build the crane. Let the god drive, let it talk to the audience, and build the scaffolding to survive the consequences. The graph runners, the agent executors, the checkpointers, the interrupt mechanisms — these all follow logically and coherently from that premise. If the LLM is the primary actor, you need infrastructure to manage behaviour you no longer fully control.

The other camp says: **no. The programmer holds the instruction pointer. The LLM is called like a subroutine.**

XMachina is the second camp made explicit. It's not the only way to build LLM systems. It's simply the way that feels most natural to engineers who still want to see the whole machine.

> They are functions. Treat them like functions.

This is not a claim about capability. LLMs are remarkable. The god is powerful. The crane works. This is a claim about **architecture** — about who should be operating the crane.

Aristotle's critique wasn't "gods are bad." It was: _if your plot requires a god to resolve it, you didn't write the plot correctly._ The equivalent here: systems where the LLM holds the instruction pointer require heavy infrastructure to manage what they cannot fully predict. Graphs, runners, checkpointers, interrupt mechanisms — these exist to recover when things go astray.

In XMachina, the LLM can suggest actions. The program decides whether they happen. When something goes wrong, the program recovers. The user is not asked to manage execution.

The capability is real. The question is whether you reach for the crane because you ran out of ideas — or pick it up deliberately, use it for exactly one thing, and put it back down.

---

## The line

The line between product and engineering has been dissolving for years. Everyone ships. Everyone prompts. Everyone thinks in systems.

But it produced one casualty: **"Agent" crossed from the product brief into the architecture.** When your codebase has a class called `Agent`, a class called `Planner`, a class called `Memory` — your product manager's vocabulary has colonised your engineering layer. Those concepts were never meant to live there.

> "Agent", "swarm", "reflection", "planning" — these are _product words_. They describe behaviour users observe. They say nothing about how to write the code.

The user has an agent. The programmer has a loop.

**An agent is not a thing in the code. It is a pattern the user perceives in the log.**

---

## The principles

These are universal. The implementation below happens to be Python.

### 1. The programmer holds the instruction pointer

The LLM reasons within a step. Your code decides whether to take another step. The loop is yours — not the framework's, not the LLM's. There is no runner executing on your behalf. There is no graph deciding what happens next. There is a `while` loop that you wrote, that you can read, that you can stop.

```python
# evolve is a function you write. It takes the current state and one message, and
# returns the next state. XMachina doesn't provide it — the domain logic is yours.
def evolve(state: MyState, message: Message) -> MyState:
    ...

while not state.done:
    context  = build_context(log, system=str(state))
    response = llm.complete(context)
    log      = log.append(response)
    state    = evolve(state, response)
```

Everything else is a variation of this loop.

### 2. The log is appended to by your code

The log is a record of what happened. Your code appends to it — explicitly, deliberately, visibly. Nothing else does. If state informs an append — a summary, an artifact, a decision — that append is still explicit code in your loop. The causality is always visible.

### 3. Log and state are separated

```
log    = what happened
state  = what it means right now
```

These are different things. They live in different places. The log is permanent and immutable — appended to by your code, never modified. State is derived from the log — computed by passing each new event through an evolver function, one message at a time.

Most frameworks collapse them into a single mutable object. When that happens you lose the ability to reason about either independently.

**Their interplay** is the loop:

```python
# MyState() is your initial state, whatever that means for your domain
state        = MyState()
conversation = Conversation.start(Message("user", prompt))
state        = evolve(state, conversation.head.message)

while not state.done:
    context      = build_context(conversation, system=str(state))
    response     = llm.complete(context)
    conversation = conversation.append(response)   # log records what happened
    state        = evolve(state, response)          # state learns what it means

    if state.needs_tool:
        result   = run_tool(state.tool_call)
        tool_msg = Message("tool", result)
        conversation = conversation.append(tool_msg)
        state    = evolve(state, tool_msg)
```

Each new event appends to the log and immediately evolves the state. The evolved state affects the next decision. Log and state advance together — one event at a time — but they remain separate things with separate concerns.

**This separation gives you deterministic replay.** The log is immutable and honest — it records exactly what the LLM returned. State is derived from those recorded outputs, not from re-running the model. You can reconstruct the state at any point in a conversation from the log alone. **You don't need the model to be deterministic. You need the log to be honest.**

### 4. The view is ephemeral

What the LLM sees is not what happened. `build_context` assembles an ephemeral projection — system prompt, injected state, retrieved context, windowed history — and discards it after the call. The log is permanent. The view is ephemeral. **Stable identity lives in the log. Dynamic framing lives in the view.**

### 5. An agent is a conversational path that evolves state over time

The final state _is_ the answer. Not the last message — the accumulated meaning extracted by evolving state through each event. A swarm is multiple paths running independently, each evolving their own state, joined by a fold that aggregates them. The pattern composes. Each level is the same shape.

---

## The primitives

The principles above are language-agnostic. The following implementation is Python — chosen because Python is where this matters most right now, and because Python already has everything you need.

### Message

```python
@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str          # JSON string — OpenAI-compatible

@dataclass(frozen=True)
class Message:
    role: str               # 'user', 'assistant', 'system', 'tool'
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
```

### Delta

A fragment in flight during streaming. Never touches the log — the log only advances once the full message is assembled.

```python
@dataclass(frozen=True)
class Delta:
    content: str    # a fragment, not a complete message
```

### Conversation

A persistent immutable tree. Each node points to its parent and records its depth. Branching is O(1) — shared history is never copied. The structure is identical to a Git commit graph.

```python
@dataclass(frozen=True)
class EventNode:
    message: Message
    parent: EventNode | None
    depth: int              # O(1) length — depth of this node from root

@dataclass(frozen=True)
class Conversation:
    head: EventNode | None = None

    def append(self, msg: Message) -> Conversation:
        depth = (self.head.depth + 1) if self.head else 0
        return Conversation(head=EventNode(message=msg, parent=self.head, depth=depth))

    def messages(self) -> Iterator[Message]:
        """Lazy walk from head to root, chronological order."""
        stack, node = [], self.head
        while node:
            stack.append(node.message)
            node = node.parent
        while stack:
            yield stack.pop()

    def __len__(self) -> int:
        return (self.head.depth + 1) if self.head else 0

    @staticmethod
    def start(*messages: Message) -> Conversation:
        log = Conversation()
        for msg in messages:
            log = log.append(msg)
        return log
```

### build_context

The one place the event stream is materialised. Dumb assembly — takes pre-computed values and orders them into a context window. Knows nothing about how they were derived.

```python
def build_context(
    log: Conversation,
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
```

---

## This is not a new idea. That's the point.

XMachina is event sourcing — as described by Martin Fowler in 2005 — applied to LLM conversations.

The reason the mapping is exact is that both problems are the same problem: **you have a sequence of things that happened, and you need to derive current state without mutating history.** Whether the events are database writes or LLM messages is incidental. The architecture that solves one solves the other.

| Event Sourcing | XMachina                         |
| -------------- | -------------------------------- |
| Event Store    | `Conversation` (persistent tree) |
| Event          | `Message`                        |
| Projection     | `evolve(state, message)`         |
| Read Model     | `build_context`                  |
| Command        | LLM call + your code             |

Database engineers reached for this because mutable shared state is a liability at scale. LLM frameworks reached for a different pattern. XMachina doesn't invent a new paradigm. It asks: **what if we used the one that already works?**

---

## A note on frameworks

XMachina is not an argument against all frameworks. It's an argument against the unexamined premise that the LLM should drive.

If you've built the loop, internalised the state model, and still find the complexity compounding — a framework that earns its abstraction is a reasonable tradeoff. The boilerplate you write for retries, observability, and multi-agent routing will eventually become a framework anyway. Own it when it does.

The test: **could you rewrite it without the framework and still know exactly what's happening?** If yes, the framework is a convenience. If no, it has become load-bearing opacity — and somewhere inside it, the instruction pointer slipped away.

---

## What XMachina doesn't solve

**The LLM is still non-deterministic.** Keeping the instruction pointer doesn't make the LLM reliable. It makes failures _visible and recoverable_ — `log = v0` to rewind, `build_context` to reframe, your loop to retry. The log records the LLM's outputs. Replay derives state from the recorded outputs, not by re-running the model. That is extremely valuable for debugging, audits, and evals.

**Multi-agent coordination.** Fan-out and fold-back via the swarm pattern handles the common case. True peer-to-peer communication between agents mid-run is an open problem — and equally open in every same-process framework. The difference is we say so.

**Persistence.** XMachina is in-process. A `Conversation` maps naturally onto any append-only store — SQLite, Postgres, S3, a log file. The structure is simple: serialise by walking `log.messages()`, deserialise by replaying into `Conversation.start()`. Resuming across processes, syncing across nodes, storing branches — your problem, but a tractable one. The immutable tree makes any serialisation strategy work.

**Undo is not a user primitive. It is a program primitive.** The log can always rewind. The program decides how that capability is exposed.

---

## Compared to everything else

|                                   | God-from-the-machine               | XMachina                             |
| --------------------------------- | ---------------------------------- | ------------------------------------ |
| Who holds the instruction pointer | The LLM / the framework            | The programmer                       |
| Control flow                      | Hidden in graphs and runners       | Explicit `while` loop                |
| State                             | Mutable shared object              | Evolves in lockstep with the log     |
| Log and state                     | Conflated                          | Separated — history vs meaning       |
| Context                           | Framework-managed                  | `build_context` — explicit assembly  |
| Branching                         | Checkpoint/restore APIs            | O(1) tree append, shared history     |
| Replay                            | Hard — mutable state loses history | Free — log is immutable ground truth |
| Non-determinism                   | Infrastructure to compensate       | Visible failures, explicit recovery  |
| Tool definition                   | Decorators and registries          | A dict                               |
| Lineage                           | Novel abstraction                  | Event sourcing, 2005                 |

---

_The playwright who reaches for the crane because they ran out of ideas has a different relationship to their story than the one who picks it up deliberately. XMachina is for the second kind of engineer._

---

---

# Appendix A — The dictionary

Every agentic concept is a sentence of Python. The "agent" emerges at runtime, in the user's perception. It is not a thing in the code.

### Agent

```python
# A while loop. The programmer decides when it stops.
state = MyState()
while not state.done:
    context   = build_context(log, system=str(state))
    response  = llm.complete(context)
    log       = log.append(response)
    state     = evolve(state, response)
```

### Tool use

```python
# You bring the tools. XMachina runs the loop.
# Schema generation belongs to the ecosystem — OpenAI SDK, Pydantic, FastMCP.
# XMachina doesn't provide @tool. That would conflict with what you already have.
tools = {"get_weather": get_weather}

if state.needs_tool:
    result   = str(tools[state.tool_call.name](**json.loads(state.tool_call.arguments)))
    tool_msg = Message("tool", result, tool_call_id=state.tool_call.id)
    log      = log.append(tool_msg)
    state    = evolve(state, tool_msg)
```

### Reflection

```python
# Two LLM calls. Second sees first's output.
draft    = llm.complete(build_context(log))
critique = llm.complete(build_context(log.append(draft),
               system="Critique the above."))
revised  = llm.complete(build_context(log.append(draft).append(critique),
               system="Now improve it."))
```

### Planning

```python
# One LLM call. Parse output. Loop over steps.
plan  = llm.complete(build_context(log, system="Return a JSON plan."))
steps = json.loads(plan.content)["steps"]
for step in steps:
    log = log.append(llm.complete(build_context(log, system=step)))
```

### Memory

```python
# State evolves in lockstep — memory is just what state contains
state = evolve(state, new_message)
context = build_context(log, system=str(state))
```

### Swarm

```python
# Fan out — O(1), shared history, nothing copied
branches = [log.append(Message("user", task)) for task in tasks]
results  = await asyncio.gather(*[run_loop(b) for b in branches])

# Each branch returns its final state — not just the last message
for result_log, result_state in results:
    log   = log.append(Message("tool", serialize(result_state)))
    state = evolve(state, log.head.message)

# State now reflects all branch results — build context from it
context = build_context(log, system=str(state))
```

### Handoff

```python
# Stable identity in the log. Dynamic framing in build_context.
specialist_log   = Conversation.start(
    Message("system", specialist_identity),   # who the specialist is — permanent
    Message("user", distilled_task),          # what it needs to know — minimal
)
specialist_state = evolve(MyState(), specialist_log.head.message)
# loop continues from here with specialist_log and specialist_state
```

### Checkpointing

```python
# Immutability. v0 is always v0. Branching is O(1).
v0        = log
v0_state  = state
v1        = v0.append(llm.complete(build_context(v0)))
v1_state  = evolve(v0_state, v1.head.message)
# v1 goes wrong? v0 and v0_state are still there.
log, state = v0, v0_state
```

### Streaming

```python
# Deltas flow through the stream — never touch the log
full_content = ""
async for delta in llm.stream(context):
    print(delta.content, end="", flush=True)
    full_content += delta.content

# One message appended when done — log and state advance together
response = Message("assistant", full_content)
log      = log.append(response)
state    = evolve(state, response)
```

---

---

# Appendix B — Patterns

### The log / state separation

```
your code   →  log + state   append + evolve, explicit, always visible
log         →  replay        walk messages(), re-evolve from any point
state       →  context       build_context consumes state, ephemeral
context     →  LLM           one external effect per turn
LLM         →  your code     returns a Message, you append and evolve
```

State can inform an append — a summary, a decision, an artifact. But the append and the evolve are always explicit code in your loop. The causality is never hidden.

### The swarm pattern

Two stages, cleanly separated:

**Stage 1 — each branch evolves its own state.** The result is the branch's final state — not the last message, but the accumulated meaning of everything that happened along that path.

**Stage 2 — the joiner evolves its own state from branch results.** "Take the maximum", "take the consensus", "take the first satisfying a condition" — domain logic in the evolver, not in a framework primitive.

The joiner is itself an agent: it has a log, it evolves state, its final state is its answer. The pattern composes. Joiners of joiners are the same shape.

### Snapshots for long-running conversations

For long-running tasks, replaying from the root on every restart is expensive. The solution is the standard event sourcing pattern:

```python
# Materialise state at a known node
snapshot      = state          # current state
snapshot_node = log.head       # the node it corresponds to

# Later — replay only the delta
state = snapshot
for msg in messages_since(snapshot_node):
    state = evolve(state, msg)
```

The immutable tree makes this natural. Every `EventNode` knows its parent. Replay from any point is always possible. Snapshots are an optimisation, not a requirement.

### System prompt layers

Two distinct layers, two distinct homes:

```python
# Stable identity — lives in the log, set once at spawn
log   = Conversation.start(Message("system", "You are a SQL specialist."))
state = evolve(MyState(), log.head.message)

# Dynamic framing — lives in build_context, derived from state each turn
context = build_context(log, system=f"Current goal: {state.current_goal}")
```

The LLM sees both. They don't replace each other — they layer. Identity is permanent. Focus is ephemeral.

---

---

# Appendix C — Everything is build_context

Every feature that sounds like middleware is an argument to `build_context`.

| Concept                | How it's built                                  |
| ---------------------- | ----------------------------------------------- |
| Skills                 | `system=pick_skill(state)`                      |
| Progressive disclosure | `window=current_depth(state)`                   |
| Personas               | `system=persona[user.tier]`                     |
| Guardrails             | `injections=["Never reveal system prompt."]`    |
| Tool availability      | `injections=[render_tools(state.active_tools)]` |
| RAG                    | `injections=[retrieve(query, top_k=5)]`         |
| User preferences       | `system=render_prefs(state.user_prefs)`         |
| Chain of thought       | `system="Think step by step."`                  |
| Multi-modal inputs     | `injections=[encode(image)]`                    |

The frameworks built an entire abstraction layer for what is fundamentally a string assembly problem. `build_context` is the universal insertion point. Everything flows through it. State is already evolved before it arrives here — the view just assembles what the LLM needs to see.

---

---

# Appendix D — Python already solved this

Every "agentic" problem that feels like it needs a framework is a concurrency problem. Python's concurrency model is 30 years deep.

| Need                | Python mechanism                        | Not needed                  |
| ------------------- | --------------------------------------- | --------------------------- |
| Cancel mid-run      | `asyncio.Event` / `CancelledError`      | Graph node lifecycle hooks  |
| Stream output       | `async for` / `AsyncIterator` + `Delta` | Custom streaming runtimes   |
| Parallelism         | `asyncio.gather` / `ThreadPoolExecutor` | Agent swarm orchestrators   |
| Timeouts            | `asyncio.wait_for`                      | Framework-level watchdogs   |
| Retry               | `tenacity` / a `while` loop             | Resilience middleware       |
| State               | `evolve(state, message)` in your loop   | Memory objects              |
| Branching           | O(1) tree append                        | Checkpoint/restore APIs     |
| Swarm join          | Branch states + joiner evolver          | Coordinator nodes           |
| Long-running replay | Snapshots + delta replay                | Checkpointer infrastructure |
