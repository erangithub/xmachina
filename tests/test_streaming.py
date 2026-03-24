from xmachina import build_context
from xmachina.llms import EchoLLM
from xmachina.environment import Environment


def test_streaming():
    llm = EchoLLM()
    env = Environment(continue_live=True)
    env.register_llm_stream_fn(llm.stream, name="llm_complete")  # event name: llm.complete
    env.register_input_fn(lambda: "")
    env.add_message("user", "hello")
    env.add_message("assistant", "echo: hello")
    env.rewind()

    env.input()  # replays user message

    full_content = ""
    for delta in env.llm_complete(build_context(env.history())):
        full_content += delta.content

    messages = list(env.history().iter_messages())
    assert len(messages) == 2
    assert messages[0].content == "hello"
    assert (messages[1].content or "").strip() == "echo: hello"


if __name__ == "__main__":
    test_streaming()
