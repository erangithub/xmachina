from xmachina import Message, EventLog, build_context
from xmachina.llms import EchoLLM
from xmachina.environment import replay

def main():
    log = EventLog()
    log = log.append(Message("user", "hello"))
    log = log.append(Message("assistant", "echo: hello"))
    env = replay(log, llm=EchoLLM(), continue_live=True)
    env.input()
    response = env.llm_complete(build_context(env.log))
    print(response)

if __name__ == "__main__":
    main()
