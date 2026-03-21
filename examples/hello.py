from xmachina import Message, EventLog, build_context
from xmachina.llms import LMStudioLLM  # or OllamaLLM
from xmachina.environment import replay

def main():
    log = EventLog()
    log = log.append(Message("user", "what is 2 plus 5"))
    log = log.append(Message("assistant", "7"))
    env = replay(log, llm=LMStudioLLM(), continue_live=True)
    env.input()
    response = env.llm_complete(build_context(env.log))
    print(response)

if __name__ == "__main__":
    main()
