"""LLM interface: one generate(prompt) -> LLMResult, with mock and Ollama backends."""
import random
import re
import zlib
from dataclasses import dataclass

FACT_PREFIX = "FACT: "
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text):
    """Remove <think>...</think> blocks (Qwen3 thinking mode) so they never reach the metrics."""
    return _THINK.sub("", text).strip()


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int


class MockLLM:
    """Keeps a random fraction of the FACT lines. Known answer for testing the metric."""

    def __init__(self, keep_fraction=0.5, seed=0):
        self.keep_fraction = keep_fraction
        self.seed = seed

    def generate(self, prompt: str) -> LLMResult:
        lines = [l for l in prompt.splitlines() if l.startswith(FACT_PREFIX)]
        rng = random.Random(self.seed + zlib.crc32(prompt.encode()))
        kept = [l[len(FACT_PREFIX):] for l in lines if rng.random() < self.keep_fraction]
        return LLMResult(" ".join(kept), prompt_tokens=len(prompt.split()))


class OllamaLLM:
    """Real backend. NOT tested against a live Ollama server in this repo: run Check A-D of the
    setup guide on the GPU machine before any experiment (GPU used, prompt token count,
    thinking off, determinism)."""

    def __init__(self, model, num_ctx, temperature, seed, think=False, client=None):
        if client is None:
            import ollama                      # imported here: the laptop needs no Ollama
            client = ollama.Client()
        self.client, self.model, self.num_ctx = client, model, num_ctx
        self.temperature, self.seed, self.think = temperature, seed, think

    def generate(self, prompt: str) -> LLMResult:
        kwargs = dict(model=self.model,
                      messages=[{"role": "user", "content": prompt}],
                      options={"num_ctx": self.num_ctx, "temperature": self.temperature,
                               "seed": self.seed})
        try:
            r = self.client.chat(think=self.think, **kwargs)
        except TypeError:                      # older ollama package without `think`
            r = self.client.chat(**kwargs)
        try:
            tokens = r["prompt_eval_count"]
        except (KeyError, TypeError):
            tokens = None
        if tokens is None:                     # can be missing when a prompt is cached
            tokens = int(len(prompt.split()) * 1.5)
            print("WARNING: prompt_eval_count missing; token count is a rough estimate")
        if tokens >= 0.9 * self.num_ctx:
            print(f"WARNING: prompt used {tokens}/{self.num_ctx} tokens, may be truncated")
        return LLMResult(strip_think(r["message"]["content"]), tokens)


def make_llm(cfg):
    c = cfg["llm"]
    if c["backend"] == "mock":
        return MockLLM(c["mock_keep_fraction"], c["seed"])
    return OllamaLLM(c["model"], c["num_ctx"], c["temperature"], c["seed"], c["think"])
