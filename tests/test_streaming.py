from xmachina import Message, EventLog, build_context
from xmachina.llms import EchoLLM
from xmachina.environment import replay


def test_streaming():
    log = EventLog()
    log = log.append(Message("user", "hello"))
    log = log.append(Message("assistant", "echo: hello"))
    env = replay(log, llm=EchoLLM(), continue_live=True)
    env.input()  # replays user message

    full_content = ""
    for delta in env.llm_stream(build_context(env.log)):
        full_content += delta.content

    messages = list(env.log.messages())
    assert len(messages) == 2
    assert messages[0].content == "hello"
    assert (messages[1].content or "").strip() == "echo: hello"