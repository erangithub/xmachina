from xmachina import Message, EventLog, build_context
from xmachina.mock import EchoLLM


def test_hello_world():
    llm = EchoLLM()
    conversation = EventLog.start(Message("user", "hello"))
    context = build_context(conversation)
    response = llm.complete(context)
    conversation = conversation.append(response)

    assert conversation.head.message.role == "assistant"
    assert conversation.head.parent.message.content == "hello"
    assert "hello" in conversation.head.message.content
    print("test_hello_world passed")


if __name__ == "__main__":
    test_hello_world()