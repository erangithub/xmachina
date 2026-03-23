from xmachina.llms import EchoLLM
from xmachina.environment import Environment


_price_counter = 0


def get_price(ticker):
    global _price_counter
    _price_counter += 1
    return {"ticker": ticker, "price": 100 + _price_counter}


def test_nondet_replay():
    global _price_counter
    _price_counter = 0

    env = Environment(llm=EchoLLM(), input_fn=lambda: "", continue_live=True)
    det_get_price = env.nondet(get_price)

    result1 = det_get_price("AAPL")
    assert result1 == {"ticker": "AAPL", "price": 101}

    env.rewind()
    result2 = det_get_price("AAPL")
    assert result2 == {"ticker": "AAPL", "price": 101}

    print("test_nondet_replay passed")


if __name__ == "__main__":
    test_nondet_replay()
