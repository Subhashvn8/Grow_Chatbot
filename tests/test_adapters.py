import sys
import types
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from grow_faq.config import load_settings
from grow_faq.llm.adapters import MAX_TOKENS, GroqLLM, LLMError


def fake_groq_module(finish_reason="stop", content="Answer."):
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            message = SimpleNamespace(content=content)
            choice = SimpleNamespace(message=message, finish_reason=finish_reason)
            return SimpleNamespace(choices=[choice])

    class Groq:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=Completions())

    module = types.ModuleType("groq")
    module.Groq = Groq
    return module, calls


class GroqAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = replace(load_settings(), llm_model="openai/gpt-oss-20b")

    def test_reasoning_model_gets_low_effort_and_room_to_answer(self) -> None:
        module, calls = fake_groq_module()
        with patch.dict(sys.modules, {"groq": module}):
            text = GroqLLM(self.settings, "key").generate("system", "user")

        self.assertEqual(text, "Answer.")
        self.assertEqual(calls[0]["reasoning_effort"], "low")
        self.assertGreaterEqual(calls[0]["max_tokens"], 2048)
        self.assertEqual(MAX_TOKENS, calls[0]["max_tokens"])

    def test_other_models_do_not_get_reasoning_effort(self) -> None:
        module, calls = fake_groq_module()
        settings = replace(self.settings, llm_model="llama-3.3-70b-versatile")
        with patch.dict(sys.modules, {"groq": module}):
            GroqLLM(settings, "key").generate("system", "user")

        self.assertNotIn("reasoning_effort", calls[0])

    def test_truncated_output_is_an_error(self) -> None:
        module, _ = fake_groq_module(finish_reason="length", content="The ratio is")
        with patch.dict(sys.modules, {"groq": module}):
            with self.assertRaises(LLMError):
                GroqLLM(self.settings, "key").generate("system", "user")


if __name__ == "__main__":
    unittest.main()
