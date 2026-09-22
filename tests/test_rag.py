from __future__ import annotations

import sys
import types

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from telecom_rag.rag import answer_with_rag, get_llm, message_text


def test_message_text_extracts_visible_content_blocks() -> None:
    content = [
        {"type": "text", "text": "First line."},
        {"type": "text", "text": "Second line."},
    ]

    assert message_text(content) == "First line.\nSecond line."


def test_cloudflare_llm_uses_precleanup_request_shape(monkeypatch) -> None:
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
    assert "extra_body" not in captured



def test_answer_with_rag_does_not_make_a_second_citation_repair_call() -> None:
    llm = FakeListChatModel(
        responses=["## Answer\nRSRP measures received reference-signal power."]
    )
    docs = [
        Document(
            page_content="RSRP is the received power of reference signals.",
            metadata={"source": "3GPP TS 38.215", "source_id": "38.215"},
        )
    ]

    result = answer_with_rag(llm, "What does RSRP measure?", docs)

    assert result["answer"].endswith("power.")
    assert result["sources"][0]["citation"] == "S1"


def test_answer_with_rag_keeps_existing_valid_citation() -> None:
    llm = FakeListChatModel(
        responses=["## Answer\nRSRP measures received reference-signal power [S1]."]
    )
    docs = [
        Document(
            page_content="RSRP is the received power of reference signals.",
            metadata={"source": "3GPP TS 38.215", "source_id": "38.215"},
        )
    ]

    result = answer_with_rag(llm, "What does RSRP measure?", docs)

    assert result["answer"].endswith("[S1].")
