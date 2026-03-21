from xmachina import Message, EventLog, build_context
#from xmachina.mock import EchoLLM
from xmachina.llms import LMStudioLLM  # or OllamaLLM

def main():
    llm = LMStudioLLM()
    conversation = EventLog.start(Message("user", "what is 2 plus 5"))
    context = build_context(conversation)
    response = llm.complete(context)
    conversation = conversation.append(response)
    print(response)

if __name__ == "__main__":
    main()
