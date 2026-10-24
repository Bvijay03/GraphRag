from llm import MockLLM, OllamaLLM, strip_think


def _prompt(n):
    return "\n".join(f"FACT: Entity{i} works_at Org{i}." for i in range(n))


def test_mock_keeps_about_half():
    out = MockLLM(0.5, seed=1).generate(_prompt(1000)).text
    assert 400 < out.count("Entity") < 600


def test_mock_is_deterministic():
    p = _prompt(50)
    assert MockLLM(0.5, 1).generate(p).text == MockLLM(0.5, 1).generate(p).text


def test_strip_think_removes_reasoning_block():
    assert strip_think("<think>\nreasoning\n</think>\nA works at B.") == "A works at B."


class StubClient:
    def __init__(self, tokens=100, content="<think>x</think>A works at B.", accept_think=True):
        self.tokens, self.content, self.accept_think = tokens, content, accept_think
        self.calls = []

    def chat(self, **kw):
        if "think" in kw and not self.accept_think:
            raise TypeError("unexpected keyword argument 'think'")
        self.calls.append(kw)
        out = {"message": {"content": self.content}}
        if self.tokens is not None:
            out["prompt_eval_count"] = self.tokens
        return out


def test_ollama_passes_settings_and_strips_think():
    c = StubClient()
    r = OllamaLLM("qwen3:14b", 16384, 0, 42, think=False, client=c).generate("hi")
    kw = c.calls[0]
    assert kw["options"] == {"num_ctx": 16384, "temperature": 0, "seed": 42}
    assert kw["think"] is False and kw["model"] == "qwen3:14b"
    assert r.text == "A works at B." and r.prompt_tokens == 100


def test_ollama_falls_back_when_think_argument_unsupported():
    c = StubClient(accept_think=False)
    assert OllamaLLM("m", 1000, 0, 1, client=c).generate("hi").text == "A works at B."


def test_ollama_warns_near_context_limit(capsys):
    OllamaLLM("m", 1000, 0, 1, client=StubClient(tokens=950)).generate("hi")
    assert "may be truncated" in capsys.readouterr().out


def test_ollama_missing_token_count_is_estimated_and_warned(capsys):
    r = OllamaLLM("m", 1000, 0, 1, client=StubClient(tokens=None)).generate("a b c d")
    assert r.prompt_tokens == 6 and "estimate" in capsys.readouterr().out
