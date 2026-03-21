from xmachina import Message, EventLog, build_context
from xmachina.llms import EchoLLM
from xmachina.environment import replay


def main():
    log = EventLog()
    log = log.append(Message("user", "And I think to myself, what a wonderful world."))
    log = log.append(Message("assistant", "echo: And I think to myself, what a wonderful world."))
    env = replay(log, llm=EchoLLM(), continue_live=True)
    env.input()

    full_content = ""
    for delta in env.llm_stream(build_context(env.log)):
        print(delta.content, end="", flush=True)
        full_content += delta.content

    print(f"\nLogged {len(env.log)} messages")


if __name__ == "__main__":
    main()