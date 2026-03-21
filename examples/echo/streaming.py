import asyncio
from xmachina import Message, Conversation, build_context
from xmachina.mock import EchoLLM


async def main():
    llm = EchoLLM()
    conversation = Conversation.start(Message("user", "And I think to myself, what a wonderful world."))
    context = build_context(conversation)

    full_content = ""
    async for delta in llm.stream(context):
        print(delta.content, end="", flush=True)
        full_content += delta.content

    # stream is done — now advance the log
    conversation = conversation.append(Message("assistant", full_content))

    print(f"\nLogged {len(conversation)} messages")


if __name__ == "__main__":
    asyncio.run(main())