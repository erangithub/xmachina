from xmachina import Message, EventLog, build_context
from xmachina.llms import EchoLLM
from xmachina.environment import replay, live


def summarize(env) -> str:
    """Run summarization in a forked branch. Fork shares parent's store."""
    sub = env.fork(parent=env)
    context = build_context(sub.log, system="Summarize the conversation.")
    response = sub.llm_complete(context)
    return f"Summary: {response.content}"

def flow(env):
    user_input = "long conversation here..."
    env.input_str(user_input)
    print(f"User: {user_input}")

    summary = summarize(env)
    print(f"Flow: {summary}")

    context = build_context(env.log, injections=[summary])
    response = env.llm_complete(context)
    print(f"Assistant: {response}")
    
    print("-" * 20)

def main():
    llm = EchoLLM()

    print("=== First run: replay + continue live ===")
    live_env = live(llm=llm, input_fn=input)
    flow(live_env)
    
    print("=== Second run: replay from saved store ===")
    replay_env = replay(live_env.log, llm=llm, input_fn=input, continue_live=False)
    flow(replay_env)


if __name__ == "__main__":
    main()

