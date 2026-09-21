from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st

from telecom_rag.bootstrap import ensure_demo_assets
from telecom_rag.config import (
    DOCS_DIR,
    MAX_QUESTION_CHARS,
    MAX_REQUEST_UNITS_PER_SESSION,
    MAX_TOP_K_PUBLIC,
    OPENAI_MODEL,
    OLLAMA_MODEL,
    PROCESSED_KPI_PATH,
    VECTOR_STORE_DIR,
)
from telecom_rag.data import load_processed_kpis
from telecom_rag.graph import build_graph
from telecom_rag.rag import get_llm, load_vector_store


st.set_page_config(
    page_title="Telecom RAG",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _read_secret(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return str(value) if value else None


# Streamlit Community Cloud secrets are copied into environment variables so the
# rest of the project can use the same code path as local development.
for secret_name in ("OPENAI_API_KEY", "OPENAI_MODEL"):
    secret_value = _read_secret(secret_name)
    if secret_value and not os.getenv(secret_name):
        os.environ[secret_name] = secret_value


@st.cache_resource(show_spinner="Preparing the reproducible demo assets...")
def cached_bootstrap():
    return ensure_demo_assets()


@st.cache_resource(show_spinner="Loading embeddings and FAISS index...")
def cached_store():
    return load_vector_store(build_if_missing=False)


@st.cache_resource(show_spinner=False)
def cached_llm(provider: str, model: str):
    return get_llm(provider=provider, model=model)


@st.cache_data(show_spinner=False)
def cached_kpis() -> pd.DataFrame | None:
    if not PROCESSED_KPI_PATH.exists():
        return None
    return load_processed_kpis()


if "request_units_used" not in st.session_state:
    st.session_state.request_units_used = 0


st.title("📡 5G Network Diagnostics RAG Assistant")
st.caption(
    "Real Ericsson/AERPAW KPI measurements + LangGraph + LangChain + FAISS + grounded LLM answers."
)

bootstrap = cached_bootstrap()

if not bootstrap.docs_ready or not bootstrap.vector_ready:
    st.error(
        "The RAG corpus or vector index could not be prepared on this server. "
        "Open the deployment details below for the exact bootstrap error."
    )
    with st.expander("Deployment bootstrap details", expanded=True):
        st.write("KPI:", bootstrap.kpi_message)
        st.write("Documents:", bootstrap.docs_message)
        st.write("FAISS:", bootstrap.vector_message)
    st.stop()

kpis = cached_kpis() if bootstrap.kpi_ready else None

openai_available = bool(os.getenv("OPENAI_API_KEY"))
remaining_units = max(0, MAX_REQUEST_UNITS_PER_SESSION - st.session_state.request_units_used)

with st.sidebar:
    st.header("Demo controls")

    if openai_available:
        provider = "openai"
        model = os.getenv("OPENAI_MODEL", OPENAI_MODEL)
        st.success("Hosted model configured")
        st.caption(f"Provider: OpenAI · Model: {model}")
    else:
        provider = st.selectbox("LLM provider", ["ollama", "openai"], index=0)
        if provider == "ollama":
            model = st.text_input("Ollama model", value=os.getenv("OLLAMA_MODEL", OLLAMA_MODEL))
            st.caption("Local/free. Start Ollama before running the app.")
        else:
            model = st.text_input("OpenAI model", value=os.getenv("OPENAI_MODEL", OPENAI_MODEL))
            st.warning("OPENAI_API_KEY is not configured.")

    use_rag = st.toggle("Use RAG", value=True)
    compare = st.toggle(
        "Also run the LLM-only baseline",
        value=False,
        help="This makes a second LLM request and therefore uses an extra request unit.",
    )
    top_k = st.slider(
        "Retrieved chunks",
        min_value=2,
        max_value=MAX_TOP_K_PUBLIC,
        value=min(4, MAX_TOP_K_PUBLIC),
    )

    st.divider()
    st.subheader("Public-demo guardrails")
    st.metric("Request units left in this session", remaining_units)
    st.caption(
        "One answer = 1 unit. Enabling the baseline comparison uses 2 units. "
        "This browser-session limit is only a convenience guardrail; the API project's "
        "hard spend limit is the real billing protection."
    )

    st.divider()
    st.subheader("Project state")
    st.write("KPI table:", "✅" if bootstrap.kpi_ready else "⚠️ docs-only")
    st.write("RAG sources:", "✅" if bootstrap.docs_ready else "❌")
    st.write("FAISS index:", "✅" if bootstrap.vector_ready else "❌")

    with st.expander("Bootstrap details"):
        st.write("KPI:", bootstrap.kpi_message)
        st.write("Documents:", bootstrap.docs_message)
        st.write("FAISS:", bootstrap.vector_message)

    with st.expander("About this project"):
        st.markdown(
            """
            **Structured data:** real Ericsson/AERPAW 5G NSA KPI measurements.

            **Starter RAG corpus:** 4 configured public sources:
            - ETSI / 3GPP TS 38.215
            - ETSI / 3GPP TS 38.214
            - AERPAW Ericsson dataset description
            - AERPAW Ericsson post-processing documentation

            The app distinguishes measured KPI evidence from retrieved technical knowledge.
            """
        )

observation = None

left, right = st.columns([1, 1])

with left:
    st.subheader("1. Select a measured KPI observation")

    if kpis is None or kpis.empty:
        st.info(
            "The KPI dataset could not be prepared on this server, so this deployment is "
            "running in documentation-only RAG mode. The RAG chatbot still works."
        )
    else:
        if "anomaly_score" in kpis.columns:
            default_df = kpis.sort_values("anomaly_score", ascending=False, na_position="last")
        else:
            default_df = kpis

        choices = default_df["observation_id"].astype(str).tolist()
        selected_id = st.selectbox("Observation", choices)
        row = kpis.loc[kpis["observation_id"].astype(str) == selected_id].iloc[0]
        observation = row.to_dict()

        show_cols = [
            c
            for c in [
                "timestamp",
                "orientation",
                "lte_rsrp_dbm",
                "nr_rsrp_dbm",
                "lte_sinr_db",
                "nr_sinr_db",
                "nr_cqi",
                "nr_mcs",
                "nr_ri",
                "throughput_mbps",
                "anomaly_score",
            ]
            if c in row.index
        ]
        st.dataframe(
            pd.DataFrame(
                {
                    "metric": show_cols,
                    "value": [row[c] for c in show_cols],
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

with right:
    st.subheader("2. Ask a question")

    default_q = (
        "Why might this observation have this throughput, and which radio measurements are "
        "most relevant to investigate?"
        if observation
        else "What do RSRP and SINR measure in a 5G NR network?"
    )

    question = st.text_area(
        "Question",
        value=default_q,
        height=130,
        max_chars=MAX_QUESTION_CHARS,
        help=f"Maximum {MAX_QUESTION_CHARS} characters in the public demo.",
    )

    units_needed = 2 if compare and use_rag else 1
    run_disabled = units_needed > remaining_units

    if run_disabled:
        st.warning("This session has reached its demo request limit.")

    run = st.button(
        "Analyze",
        type="primary",
        use_container_width=True,
        disabled=run_disabled,
    )


if run:
    cleaned_question = question.strip()

    if not cleaned_question:
        st.error("Enter a question first.")
        st.stop()

    if len(cleaned_question) > MAX_QUESTION_CHARS:
        st.error(f"Question must be at most {MAX_QUESTION_CHARS} characters.")
        st.stop()

    # Reserve the units before the API call. A failed request can still consume provider
    # resources, so the conservative choice is not to refund it automatically.
    st.session_state.request_units_used += units_needed

    try:
        store = cached_store()
        llm = cached_llm(provider, model)
        graph = build_graph(llm, store, reference_df=kpis, top_k=top_k)

        with st.spinner("Running the LangGraph workflow..."):
            result = graph.invoke(
                {
                    "question": cleaned_question,
                    "use_rag": use_rag,
                    "observation": observation,
                }
            )

        st.subheader("Answer")
        st.markdown(result["answer"])

        meta_cols = st.columns(4)
        meta_cols[0].metric("Route", result.get("route", "-"))
        meta_cols[1].metric("RAG", "On" if use_rag else "Off")
        meta_cols[2].metric("Retrieved chunks", len(result.get("sources", [])))
        meta_cols[3].metric("Latency", f"{result.get('latency_s', 0):.2f} s")

        if result.get("kpi_context"):
            with st.expander("Data-derived KPI context sent to the LLM"):
                st.code(result["kpi_context"])

        if result.get("sources"):
            st.subheader("Retrieved sources")
            for source in result["sources"]:
                page = f" — page {source['page']}" if source.get("page") else ""
                with st.expander(f"[{source['citation']}] {source['source']}{page}"):
                    st.write(source["excerpt"])

        if compare and use_rag:
            baseline_graph = build_graph(llm, store, reference_df=kpis, top_k=top_k)
            with st.spinner("Running the same LLM without retrieval..."):
                baseline = baseline_graph.invoke(
                    {
                        "question": cleaned_question,
                        "use_rag": False,
                        "observation": observation,
                    }
                )

            st.subheader("Same LLM without RAG")
            st.markdown(baseline["answer"])
            st.caption(f"Latency: {baseline.get('latency_s', 0):.2f} s")

    except Exception as exc:
        st.error(
            "The request failed. If this is the public deployment, common causes are an "
            "exhausted API balance, a project spend limit, or a temporary provider error."
        )
        with st.expander("Technical error"):
            st.exception(exc)


st.divider()
st.caption(
    "KPI percentiles and anomaly scores are relative to this measurement dataset, not "
    "universal telecom quality thresholds. Retrieved sources are shown so the technical "
    "explanation can be inspected."
)
