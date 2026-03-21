from xmachina import Message, EventLog, build_context
from xmachina.mock import EchoLLM

def main():
    llm = EchoLLM()
    conversation = EventLog.start(Message("user", "hello"))
    context = build_context(conversation)
    response = llm.complete(context)
    conversation = conversation.append(response)
    print(response)

if __name__ == "__main__":
    main()
