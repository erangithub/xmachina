from xmachina import Message, EventLog, build_context
from xmachina.llms import EchoLLM
from xmachina.environment import replay, live

def test_hello_world_replay():
    # Pre-fill log and prepare environment
    log = EventLog()
    log = log.append(Message("user", "hello"))
    log = log.append(Message("assistant", "echo: hello"))
    env = replay(log, llm=EchoLLM(), input_fn=input, continue_live=True)
    
    # Single turn run
    env.input()  # replays user message
    response = env.llm_complete(build_context(env.log))
    
    assert response.role == "assistant"
    assert response.content == "echo: hello"
    print("test_hello_world_full_replay passed")


def test_hello_world_replay_then_live():
    # Pre-fill log and prepare environment
    log = EventLog()
    log = log.append(Message("user", "hello"))
    env = replay(log, llm=EchoLLM(), input_fn=input, continue_live=True)
    
    # Single turn run
    env.input()  # replays user message
    response = env.llm_complete(build_context(env.log))

    assert response.role == "assistant"
    assert response.content == "echo: hello"
    print("test_hello_world_replay_then_live passed")


def test_hello_world_live():
    # Pre-fill log and prepare environment
    env = live(llm=EchoLLM(), input_fn=lambda: "hello")
    
    # Single turn run
    env.input()  # reads user message
    response = env.llm_complete(build_context(env.log))

    assert response.role == "assistant"
    assert response.content == "echo: hello"
    print("test_hello_world_live passed")


if __name__ == "__main__":
    test_hello_world_replay()
    test_hello_world_replay_then_live()
    test_hello_world_live()