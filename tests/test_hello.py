from xmachina import Message, EventLog, build_context
from xmachina.llms import EchoLLM
from xmachina.environment import replay


def test_hello_world():
    log = EventLog()
    log = log.append(Message("user", "hello"))
    log = log.append(Message("assistant", "echo: hello"))
    env = replay(log, llm=EchoLLM(), continue_live=True)
    env.input()  # replays user message
    response = env.llm_complete(build_context(env.log))
    assert response.role == "assistant"
    assert response.content == "echo: hello"
    print("test_hello_world passed")


if __name__ == "__main__":
    test_hello_world()