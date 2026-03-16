import pytest
from xmachina import Message, Conversation, build_context
from xmachina.mock import EchoLLM


@pytest.mark.asyncio
async def test_streaming():
    llm = EchoLLM()
    conversation = Conversation.start(Message("user", "hello"))
    context = build_context(conversation)

    full_content = ""
    async for delta in llm.stream(context):
        full_content += delta.content

    conversation = conversation.append(Message("assistant", full_content))

    messages = list(conversation.messages())
    assert len(messages) == 2
    assert messages[0].content == "hello"
    assert messages[1].content.strip() == "echo: hello"