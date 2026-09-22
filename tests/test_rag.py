from __future__ import annotations

import sys
import types

from telecom_rag.rag import get_llm, message_text


def test_message_text_extracts_visible_content_blocks() -> None:
    content = [
        {"type": "text", "text": "First line."},
        {"type": "text", "text": "Second line."},
    ]

    assert message_text(content) == "First line.\nSecond line."


def test_cloudflare_llm_disables_thinking(monkeypatch) -> None:
    captured: dict = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_module = types.ModuleType("langchain_openai")
    fake_module.ChatOpenAI = FakeChatOpenAI
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_module)
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "account")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "token")

    get_llm(
        provider="cloudflare",
        model="@cf/google/gemma-4-26b-a4b-it",
        max_output_tokens=128,
    )

    assert captured["max_tokens"] == 128
    assert captured["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
