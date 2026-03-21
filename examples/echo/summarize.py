from xmachina import Message, EventLog, build_context, DUMMY
from xmachina.llms import EchoLLM
from xmachina.environment import live, replay


def summarize(env, llm) -> str:
    """Run summarization in a forked branch. Fork shares parent's store."""
    sub = env.fork(parent=env)
    context = build_context(sub.log, system="Summarize the conversation.")
    response = sub.llm_complete(context)
    return f"Summary: {response.content}"


def main():
    llm = EchoLLM()

    print("=== First run: live summarize ===")
    env = live(llm=llm, input_fn=input)
    env._write(Message("user", "long conversation here..."))

    summary = summarize(env, llm)
    print(f"{summary}")

    print(f"Main log head: {env.log.head_id}")
    print(f"Main log messages: {list(env.log.messages())}")
    print(f"Store nodes: {list(env.log.store.nodes.keys())}")

    print()
    print("=== Second run: replay from saved store ===")
    saved_log = env.log

    env2 = replay(saved_log, llm=llm, input_fn=input, continue_live=True)
 
    print(f"Replay User: {env2.input()}")
    summary2 = summarize(env2, llm)    
    print(f"Replay: {summary2}")



if __name__ == "__main__":
    main()

