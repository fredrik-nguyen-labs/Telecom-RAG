from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st

from telecom_rag.config import DOCS_DIR, PROCESSED_KPI_PATH, VECTOR_STORE_DIR
from telecom_rag.data import load_processed_kpis
from telecom_rag.graph import build_graph
from telecom_rag.rag import get_llm, load_vector_store

# Streamlit Community Cloud stores secrets in st.secrets rather than a checked-in file.
# Local development may have no secrets.toml at all, so access defensively.
try:
    openai_key = st.secrets.get("OPENAI_API_KEY")
    openai_model = st.secrets.get("OPENAI_MODEL")
except Exception:
    openai_key = None
    openai_model = None

if openai_key and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = str(openai_key)
if openai_model and not os.getenv("OPENAI_MODEL"):
    os.environ["OPENAI_MODEL"] = str(openai_model)

st.set_page_config(page_title="Telecom RAG", page_icon="📡", layout="wide")


@st.cache_resource(show_spinner="Loading embeddings and FAISS index...")
def cached_store():
    return load_vector_store()


@st.cache_data(show_spinner=False)
def cached_kpis() -> pd.DataFrame | None:
    if not PROCESSED_KPI_PATH.exists():
        return None
    return load_processed_kpis()


st.title("📡 5G Network Diagnostics RAG Assistant")
st.caption(
    "A portfolio demo combining real Ericsson/AERPAW KPI measurements, LangGraph orchestration, "
    "LangChain retrieval, FAISS, and grounded LLM answers."
)

with st.sidebar:
    st.header("Model")
    provider = st.selectbox("LLM provider", ["ollama", "openai"], index=0)
    if provider == "ollama":
        model = st.text_input("Ollama model", value=os.getenv("OLLAMA_MODEL", "qwen3:4b"))
        st.caption("Free/local. Start Ollama and run `ollama pull qwen3:4b` first.")
    else:
        model = st.text_input(
            "OpenAI model", value=os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
        )
        if not os.getenv("OPENAI_API_KEY"):
            st.warning("OPENAI_API_KEY is not set in this process / Streamlit secrets.")

    use_rag = st.toggle("Use RAG", value=True)
    compare = st.toggle("Also run LLM-only baseline", value=False)
    top_k = st.slider("Retrieved chunks", min_value=2, max_value=8, value=4)

    st.divider()
    st.header("Project state")
    st.write("KPI table:", "✅" if PROCESSED_KPI_PATH.exists() else "❌")
    st.write("Documents:", "✅" if DOCS_DIR.exists() and any(DOCS_DIR.glob("*")) else "❌")
    st.write("FAISS index:", "✅" if (VECTOR_STORE_DIR / "index.faiss").exists() else "auto-build")

kpis = cached_kpis()
observation = None

left, right = st.columns([1, 1])
with left:
    st.subheader("1. Select a measured KPI observation")
    if kpis is None or kpis.empty:
        st.info(
            "No processed KPI table yet. Run `python scripts/download_data.py` and notebook "
            "`01_data_processing_and_eda.ipynb`. Documentation-only RAG still works."
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
            c for c in [
                "timestamp", "orientation", "lte_rsrp_dbm", "nr_rsrp_dbm",
                "lte_sinr_db", "nr_sinr_db", "nr_cqi", "nr_mcs", "nr_ri",
                "throughput_mbps", "anomaly_score",
            ] if c in row.index
        ]
        st.dataframe(pd.DataFrame(row[show_cols], columns=[selected_id]), use_container_width=True)

with right:
    st.subheader("2. Ask a question")
    default_q = (
        "Why might this observation have this throughput, and which radio measurements are "
        "most relevant to investigate?"
        if observation
        else "What do RSRP and SINR measure in a 5G NR network?"
    )
    question = st.text_area("Question", value=default_q, height=130)
    run = st.button("Analyze", type="primary", use_container_width=True)

if run:
    try:
        store = cached_store()
        llm = get_llm(provider=provider, model=model)
        graph = build_graph(llm, store, reference_df=kpis, top_k=top_k)

        with st.spinner("Running LangGraph workflow..."):
            result = graph.invoke(
                {"question": question, "use_rag": use_rag, "observation": observation}
            )

        st.subheader("Answer")
        st.markdown(result["answer"])
        meta_cols = st.columns(3)
        meta_cols[0].metric("Route", result.get("route", "-"))
        meta_cols[1].metric("RAG", "On" if use_rag else "Off")
        meta_cols[2].metric("Latency", f"{result.get('latency_s', 0):.2f} s")

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
            with st.spinner("Running the same LLM without RAG..."):
                baseline = baseline_graph.invoke(
                    {"question": question, "use_rag": False, "observation": observation}
                )
            st.subheader("Same LLM without RAG")
            st.markdown(baseline["answer"])
            st.caption(f"Latency: {baseline.get('latency_s', 0):.2f} s")

    except Exception as exc:
        st.error(str(exc))
        st.exception(exc)

st.divider()
st.caption(
    "Important: KPI percentile statements are relative to this measurement dataset. "
    "The app does not label universal radio thresholds unless a retrieved source supports them."
)
