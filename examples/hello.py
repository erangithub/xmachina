from noagent import Message, Conversation, build_context, EchoLLM


def main():
    llm = EchoLLM()
    conversation = Conversation.start(Message("user", "hello"))
    context = build_context(conversation)
    response = llm.complete(context)
    conversation = conversation.append(response)
    print(response)

if __name__ == "__main__":
    main()
