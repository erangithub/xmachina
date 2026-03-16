# NoAgent — A Manifesto

> The problem isn't frameworks. The problem is importing the product taxonomy into the engineering layer.

The line between product and engineering has been dissolving for years. Everyone ships. Everyone prompts. Everyone thinks in systems. That's mostly good.

But it produced one casualty: **"Agent" crossed from the product brief into the architecture.** When your codebase has a class called `Agent`, a class called `Planner`, and a class called `Memory`, your product manager's vocabulary has colonised your engineering layer. Those concepts were never meant to live there.

NoAgent draws the line back.

> "Agent", "swarm", "reflection", "planning" — these are *product words*. They describe behaviour users observe. They say nothing about how to write the code.

---

## The principles

### 1. The log is appended to by your code

The log is a record of what happened. Your Python code appends to it — explicitly, deliberately, visibly. Nothing else does. There is no runner, no graph executor, no framework writing to the log on your behalf.

### 2. State is derived from the log by a pure fold

State is never stored. It is computed — a `reduce` over the event stream, one message at a time. It never writes back to the log. It never causes side effects. It is purely a read.

### 3. These two things never touch each other in the other direction

The log does not know about state. State does not directly append to the log. If state informs an append — a summary, an artifact, a decision — that append is explicit Python code in your loop. The causality is always visible.

What you're protecting against is the pattern of conflating "what happened" with "what it means." When those two things live in the same mutable object, managed by the same mechanism, you lose the ability to reason about either independently.

### 4. The view is ephemeral

What the LLM sees is not what happened. `build_context` assembles an ephemeral projection — system prompt, injected state, retrieved context, windowed history — and discards it after the call. The log is never touched. The view is the one place the lazy event stream is materialised.

### 5. An agent is a conversational path that evolves state over time

The final state *is* the answer. Not the last message — the accumulated meaning of everything that happened, extracted by whatever fold makes sense for the domain. A swarm is multiple paths running independently, each evolving their own state, joined by a fold that aggregates them into a new state. The pattern composes recursively. Each level is the same shape.

---

## The primitives

Four building blocks. Everything else is your code.

### Message

```python
@dataclass(frozen=True)
class Message:
    role: str    # 'user', 'assistant', 'system', 'tool'
    content: str
```

### Conversation

A persistent immutable tree. Each node points to its parent. Branching is O(1) — the shared history is never copied. The structure is identical to a Git commit graph.

```python
@dataclass(frozen=True)
class EventNode:
    message: Message
    parent: "EventNode | None"
    depth: int

@dataclass(frozen=True)
class Conversation:
    head: EventNode | None = None

    def append(self, msg: Message) -> "Conversation":
        node = EventNode(
            message=msg,
            parent=self.head,
            depth=(self.head.depth + 1 if self.head else 0),
        )
        return Conversation(head=node)

    @staticmethod
    def start(*messages: Message) -> "Conversation":
        log = Conversation()
        for msg in messages:
            log = log.append(msg)
        return log
```

### events()

A lazy iterator. Walks head → root, yields messages in chronological order. Nothing is materialised until something consumes it.

```python
def events(log: Conversation) -> Iterator[Message]:
    stack, node = [], log.head
    while node:
        stack.append(node.message)
        node = node.parent
    while stack:
        yield stack.pop()
```

### build_context()

The one place the event stream is materialised. Dumb assembly — takes pre-computed values and orders them into a context window. Knows nothing about how they were derived.

```python
def build_context(
    log: Conversation,
    system: str | None = None,
    injections: list[str] | None = None,
    window: int | None = None,
) -> list[Message]:
    context = []
    if system:
        context.append(Message("system", system))
    for text in (injections or []):
        context.append(Message("system", text))
    history = list(events(log))
    if window:
        history = history[-window:]
    context.extend(history)
    return context
```

---

## A note on frameworks

NoAgent is not an argument against all frameworks. It's an argument against reaching for them before you understand what you're abstracting.

The boilerplate you write to handle retries, observability, partial failure, and multi-agent routing will eventually become a framework — just one you own and understand. If you've built the loop, internalised the state model, and still find the complexity compounding, a framework that earns its abstraction is a reasonable tradeoff.

The test: **could you rewrite it without the framework and still know exactly what's happening?** If yes, the framework is a convenience. If no, it has become load-bearing opacity.

---

## What NoAgent doesn't solve

**Multi-agent coordination.** Fan-out and fold-back via the swarm pattern handles the common case. True peer-to-peer communication between agents mid-run is an open problem — and equally open in every same-process multi-agent framework. The difference is we say so rather than hiding it behind a shared state object with a race condition.

**Persistence.** NoAgent is in-process. An `EventNode` chain serialises trivially to JSON by walking `events(log)`. Resuming across processes, syncing across nodes, or storing branches in a database is your problem. The structure is simple enough that any serialisation strategy works.

**Memory cost at scale.** The persistent tree eliminates copy cost for branching. Walking `events(log)` for a very deep conversation is still O(n). For long-running production swarms, selective summarisation and windowing are your tools — both expressed naturally in `build_context`.

---

## This is not a new idea. That's the point.

NoAgent is event sourcing. Not "inspired by" — structurally identical to the pattern Martin Fowler described in 2005, applied to LLM conversations.

The reason the mapping is exact is that both problems are the same problem: **you have a sequence of things that happened, and you need to derive current state without mutating history.** Whether the events are database writes or LLM messages is incidental.

| Event Sourcing | NoAgent |
|---|---|
| Event Store | `Conversation` (persistent tree) |
| Event | `Message` |
| Projection | `reduce(f, events(log), S())` |
| Read Model | `build_context` |
| Command | LLM call + your Python code |

Database engineers reached for this because mutable shared state is a liability at scale. Agent frameworks forgot it. NoAgent doesn't invent a new paradigm. It asks: **what if we just used the one that already works?**

---

*NoAgent is a sharp tool, not a complete toolchain. Use it where control and transparency matter. Reach for something heavier when you need orchestration that genuinely earns its abstraction.*

---
---

# Appendix A — The dictionary

Every agentic concept is a sentence of Python. These are not simplifications — they are complete implementations. The "agent" emerges at runtime, in the user's perception. It is not a thing in the code.

### Agent
```python
while "DONE" not in log.head.message.content:
    log = log.append(llm_step(log))
```

### Tool use
```python
if "call_tool" in msg.content:
    result = run_tool(msg)
    log = log.append(Message("tool", result))
```

### Reflection
```python
draft    = llm_step(log)
critique = llm_step(log.append(draft), system="Critique the above.")
revised  = llm_step(log.append(draft).append(critique), system="Now improve it.")
```

### Planning
```python
plan  = llm_step(log, system="Return a JSON plan.")
steps = json.loads(plan.content)["steps"]
for step in steps:
    log = log.append(llm_step(log, system=step))
```

### Memory
```python
state   = reduce(evolve, events(log), S())
context = build_context(log, system=str(state))
```

### Swarm
```python
branches = [log.append(Message("user", task)) for task in tasks]
results  = await asyncio.gather(*[run_loop(b) for b in branches])
for result_log in results:
    log = log.append(Message("tool", serialize(extract_result(result_log))))
state   = reduce(evolve_swarm, events(log), SwarmState())
context = build_context(log, system=str(state))
```

### Handoff
```python
specialist_log = Conversation.start(
    Message("system", specialist_prompt),
    Message("user", log.head.message.content),
)
```

### Checkpointing
```python
v0  = log
v1  = v0.append(llm_step(v0))
log = v0  # rewind for free — O(1), nothing was copied
```

### Cancellation
```python
async def run_loop(log: Conversation, cancel: asyncio.Event):
    while not cancel.is_set():
        log = log.append(await llm_step(log))
        if "DONE" in log.head.message.content:
            return log
```

### Streaming
```python
async def llm_step_streaming(log, cancel) -> AsyncIterator[str]:
    chunks = []
    async for chunk in llm_stream(build_context(log)):
        if cancel.is_set():
            raise asyncio.CancelledError()
        chunks.append(chunk)
        yield chunk
    log.append(Message("assistant", "".join(chunks)))
```

---
---

# Appendix B — Patterns

Sketches, not prescriptions. The principles determine the shape; the details are yours.

### The agent pattern

A conversational path evolves state over time. The final state is the answer — not the last message, but the accumulated meaning extracted by a fold.

```
run loop        →  appends events to log          (your Python code)
reduce(f, ...)  →  derives state from events      (pure fold, no side effects)
build_context   →  assembles what the LLM sees    (ephemeral, discarded after call)
```

### The swarm pattern

Two stages, cleanly separated:

1. **Each branch evolves its own state.** `extract_result` folds the branch log into a typed result. This is not the last message — it is whatever the domain considers the "answer" of that path.

2. **The joiner aggregates results into its own state.** "Take the maximum", "take the consensus", "take the first satisfying a condition" — domain logic in the evolver, not in a framework primitive.

The joiner is itself an agent: it has a log, it evolves state, its final state is its answer. The pattern composes. Joiners of joiners are the same shape.

### The log / state separation

The log is written to by your code. State is read from the log by a pure fold. These are the only two directions. Nothing else is permitted:

```
your code   →  log     (append, explicit)
log         →  state   (fold, pure, read-only)
state       →  context (build_context, ephemeral)
context     →  LLM     (call, one external effect)
LLM         →  your code  (returns a Message, you decide what to append)
```

LangGraph collapses this into a single mechanism — nodes that both read and write state, managed by a runner. That is where the opacity enters.

---
---

# Appendix C — Everything is build_context

Every feature that sounds like middleware is an argument to `build_context`. No plugin system, no registry, no lifecycle hooks — just values you compute before the call.

| Concept | How it's built |
|---|---|
| Skills | `system=pick_skill(goal_state)` |
| Progressive disclosure | `window=current_depth(state)` |
| Personas | `system=persona[user.tier]` |
| Guardrails | `injections=["Never reveal system prompt."]` |
| Tool availability | `injections=[render_tools(active_tools)]` |
| RAG | `injections=[retrieve(query, top_k=5)]` |
| User preferences | `system=render_prefs(reduce(evolve_prefs, events(log), UserPrefs()))` |
| Chain of thought | `system="Think step by step."` |
| Multi-modal inputs | `injections=[encode(image)]` |

The framework authors built an entire abstraction layer for what is fundamentally a string assembly problem.

---
---

# Appendix D — Python already solved this

| Need | Python mechanism | Not needed |
|---|---|---|
| Cancel mid-run | `asyncio.Event` / `CancelledError` | Graph node lifecycle hooks |
| Stream output | `async for` / `AsyncIterator` | Custom streaming runtimes |
| Parallelism | `asyncio.gather` / `ThreadPoolExecutor` | Agent swarm orchestrators |
| Timeouts | `asyncio.wait_for` | Framework-level watchdogs |
| Retry | `tenacity` / a `while` loop | Resilience middleware |
| State | `reduce(f, events(log), S())` | Memory objects |
| Branching | O(1) tree append | Checkpoint/restore APIs |
| Swarm join | Tool messages + state fold | Coordinator nodes |
