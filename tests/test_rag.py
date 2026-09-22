from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from telecom_rag.rag import CloudflareWorkersAIChat, answer_with_rag, format_context, get_llm


class FakeResponse:
    ok = True
    status_code = 200
    text = ""

    def __init__(self, payload: dict):
        self._payload = payload

    def json(self) -> dict:
        return self._payload




def test_cloudflare_factory_returns_direct_adapter(monkeypatch) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "account")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "token")

    llm = get_llm(
        provider="cloudflare",
        model="@cf/google/gemma-4-26b-a4b-it",
        max_output_tokens=128,
    )

    assert isinstance(llm, CloudflareWorkersAIChat)
    assert llm.model == "@cf/google/gemma-4-26b-a4b-it"
    assert llm.max_output_tokens == 128

def test_cloudflare_adapter_uses_documented_request_shape(monkeypatch) -> None:
    captured: dict = {}

    def fake_post(url, *, headers, json, timeout):
        captured.update(
            {
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            }
        )
        return FakeResponse(
            {
                "model": "@cf/google/gemma-4-26b-a4b-it",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Visible answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3},
            }
        )

    monkeypatch.setattr("telecom_rag.rag.requests.post", fake_post)
    llm = CloudflareWorkersAIChat(
        account_id="account",
        api_token="token",
        model="@cf/google/gemma-4-26b-a4b-it",
        max_output_tokens=128,
    )

    response = llm.invoke(
        [
            SystemMessage(content="You are helpful."),
            HumanMessage(content="What is RSRP?"),
        ]
    )

    assert response.content == "Visible answer"
    assert captured["json"]["max_completion_tokens"] == 128
    assert "max_tokens" not in captured["json"]
    assert captured["json"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert captured["json"]["messages"][1] == {
        "role": "user",
        "content": "What is RSRP?",
    }


def test_cloudflare_adapter_reports_empty_content_with_finish_reason(monkeypatch) -> None:
    def fake_post(url, *, headers, json, timeout):
        return FakeResponse(
            {
                "model": "@cf/google/gemma-4-26b-a4b-it",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": ""},
                        "finish_reason": "length",
                    }
                ],
                "usage": {"completion_tokens": 128},
            }
        )

    monkeypatch.setattr("telecom_rag.rag.requests.post", fake_post)
    llm = CloudflareWorkersAIChat(
        account_id="account",
        api_token="token",
        model="@cf/google/gemma-4-26b-a4b-it",
        max_output_tokens=128,
    )

    try:
        llm.invoke([HumanMessage(content="test")])
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected empty-content response to fail")

    assert "finish_reason=length" in message
    assert "completion_tokens" in message




def test_format_context_adds_canonical_source_url() -> None:
    docs = [
        Document(
            page_content="NR reference-signal measurement definition.",
            metadata={
                "source": "etsi_ts_138215_v18_5_0.pdf",
                "source_id": "etsi_ts_138215_v18_5_0",
                "page": 12,
            },
        )
    ]

    _, sources = format_context(docs)

    assert sources[0]["url"].startswith("https://www.etsi.org/")
    assert "38.215" in sources[0]["title"]
    assert sources[0]["page"] == 12

def test_answer_with_rag_returns_model_content_and_sources() -> None:
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

    assert "[S1]" in result["answer"]
    assert result["sources"][0]["citation"] == "S1"
