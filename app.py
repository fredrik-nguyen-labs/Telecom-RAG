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
    CLOUDFLARE_MODEL,
    MAX_QUESTION_CHARS,
    MAX_REQUEST_UNITS_PER_SESSION,
    MAX_TOP_K_PUBLIC,
    OPENAI_MODEL,
    OLLAMA_MODEL,
    PROCESSED_KPI_PATH,
)
from telecom_rag.data import load_processed_kpis
from telecom_rag.graph import build_graph
from telecom_rag.rag import get_llm, load_advanced_retriever
from telecom_rag.supabase_backend import (
    get_cloudflare_usage_today,
    get_supabase_status,
    load_kpis_from_supabase,
    load_supabase_retriever,
    record_cloudflare_usage,
    supabase_runtime_configured,
)


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
for secret_name in (
    "CLOUDFLARE_ACCOUNT_ID",
    "CLOUDFLARE_API_TOKEN",
    "CLOUDFLARE_MODEL",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "USE_SUPABASE",
    "SUPABASE_URL",
    "SUPABASE_PUBLISHABLE_KEY",
    "SUPABASE_ANON_KEY",
):
    secret_value = _read_secret(secret_name)
    if secret_value and not os.getenv(secret_name):
        os.environ[secret_name] = secret_value


@st.cache_resource(show_spinner="Preparing local demo assets...")
def cached_bootstrap():
    return ensure_demo_assets()


@st.cache_resource(show_spinner="Loading retrieval backend...")
def cached_retriever(backend: str):
    if backend == "supabase":
        return load_supabase_retriever()
    return load_advanced_retriever(build_if_missing=False)


@st.cache_resource(show_spinner=False)
def cached_llm(provider: str, model: str):
    return get_llm(provider=provider, model=model)


@st.cache_data(show_spinner=False)
def cached_kpis(backend: str) -> pd.DataFrame | None:
    if backend == "supabase":
        df = load_kpis_from_supabase()
        return None if df.empty else df

    if not PROCESSED_KPI_PATH.exists():
        return None
    return load_processed_kpis()


if "request_units_used" not in st.session_state:
    st.session_state.request_units_used = 0


st.title("📡 5G Network Diagnostics RAG Assistant")
st.caption(
    "Real Ericsson/AERPAW KPI measurements + LangGraph + hybrid retrieval + "
    "Supabase/pgvector or local FAISS + grounded LLM answers."
)


# Prefer persistent Supabase storage when configured and seeded. If credentials are
# missing, the migration was not applied, or the database is empty, retain the fully
# reproducible local FAISS fallback.
supabase_status: dict = {}
supabase_error: str | None = None
using_supabase = False

if supabase_runtime_configured():
    try:
        supabase_status = get_supabase_status()
        using_supabase = int(supabase_status.get("document_chunks_current", 0)) > 0
    except Exception as exc:
        supabase_error = str(exc)

storage_backend = "supabase" if using_supabase else "local"

if using_supabase:
    bootstrap = None
    kpis = cached_kpis("supabase")
else:
    bootstrap = cached_bootstrap()

    if not bootstrap.docs_ready or not bootstrap.vector_ready:
        st.error(
            "Neither the hosted Supabase RAG store nor the local RAG assets are ready."
        )
        if supabase_runtime_configured():
            st.info(
                "Supabase is configured but not seeded/available. Apply the SQL migration "
                "and run python scripts/sync_supabase.py, or fix the local bootstrap."
            )
        with st.expander("Deployment/bootstrap details", expanded=True):
            if supabase_error:
                st.write("Supabase:", supabase_error)
            elif supabase_runtime_configured():
                st.write("Supabase status:", supabase_status)
            st.write("KPI:", bootstrap.kpi_message)
            st.write("Documents:", bootstrap.docs_message)
            st.write("Local FAISS:", bootstrap.vector_message)
        st.stop()

    kpis = cached_kpis("local") if bootstrap.kpi_ready else None


cloudflare_available = bool(
    os.getenv("CLOUDFLARE_ACCOUNT_ID") and os.getenv("CLOUDFLARE_API_TOKEN")
)
openai_available = bool(os.getenv("OPENAI_API_KEY"))
remaining_units = max(
    0, MAX_REQUEST_UNITS_PER_SESSION - st.session_state.request_units_used
)


def _render_cloudflare_quota(placeholder, model_name: str) -> None:
    with placeholder.container():
        if not cloudflare_available:
            return
        st.subheader("Workers AI free quota")
        if not using_supabase:
            st.caption(
                "Quota estimate needs the Supabase usage migration and hosted backend."
            )
            return
        try:
            usage = get_cloudflare_usage_today(model_name)
            remaining = usage.get("estimated_remaining_neurons")
            used = usage.get("estimated_neurons")
            if remaining is None or used is None:
                st.caption(
                    "No Neuron conversion is configured for this model. "
                    "Check Cloudflare's Workers AI dashboard for authoritative usage."
                )
                return
            st.metric(
                "Estimated remaining today",
                f"{remaining:,.0f} Neurons",
                delta=f"{used:,.1f} app-estimated used",
                delta_color="inverse",
            )
            progress = min(max(float(used) / 10_000.0, 0.0), 1.0)
            st.progress(progress)
            st.caption(
                f"Tracked app calls today: {int(usage.get('calls', 0))}. "
                "Estimate resets at 00:00 UTC. Cloudflare dashboard is authoritative "
                "for account-wide usage."
            )
        except Exception:
            st.caption(
                "Usage meter not initialized yet. Apply the Cloudflare usage Supabase "
                "migration; inference itself can still work."
            )


with st.sidebar:
    st.header("Demo controls")

    if cloudflare_available:
        provider = "cloudflare"
        model = os.getenv("CLOUDFLARE_MODEL", CLOUDFLARE_MODEL)
        st.success("Free hosted LLM configured")
        st.caption(f"Provider: Cloudflare Workers AI · Model: {model}")
    elif openai_available:
        provider = "openai"
        model = os.getenv("OPENAI_MODEL", OPENAI_MODEL)
        st.success("Hosted LLM configured")
        st.caption(f"Provider: OpenAI · Model: {model}")
    else:
        provider = st.selectbox(
            "LLM provider", ["ollama", "cloudflare", "openai"], index=0
        )
        if provider == "ollama":
            model = st.text_input(
                "Ollama model", value=os.getenv("OLLAMA_MODEL", OLLAMA_MODEL)
            )
            st.caption("Local/free. Start Ollama before running the app.")
        elif provider == "cloudflare":
            model = st.text_input(
                "Cloudflare model",
                value=os.getenv("CLOUDFLARE_MODEL", CLOUDFLARE_MODEL),
            )
            st.warning(
                "Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN to use "
                "Workers AI."
            )
        else:
            model = st.text_input(
                "OpenAI model", value=os.getenv("OPENAI_MODEL", OPENAI_MODEL)
            )
            st.warning("OPENAI_API_KEY is not configured.")

    use_rag = st.toggle("Use RAG", value=True)
    retrieval_mode = st.selectbox(
        "Retrieval pipeline",
        ["reranked", "hybrid", "dense"],
        index=0,
        help=(
            "Local: BGE/FAISS + BM25 + RRF. Supabase: pgvector + Postgres FTS + RRF. "
            "Reranked mode applies the same cross-encoder after either backend."
        ),
    )
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

    quota_placeholder = st.empty()
    _render_cloudflare_quota(
        quota_placeholder,
        os.getenv("CLOUDFLARE_MODEL", CLOUDFLARE_MODEL),
    )

    st.divider()
    st.subheader("Public-demo guardrails")
    st.metric("Request units left in this session", remaining_units)
    st.caption(
        "One answer = 1 unit. Enabling the baseline comparison uses 2 units. "
        "This is a browser-session convenience guardrail. Cloudflare Workers AI's "
        "free allocation is enforced separately by Cloudflare."
    )

    st.divider()
    st.subheader("Project state")
    if using_supabase:
        st.success("Storage: Supabase Postgres + pgvector")
        st.write(
            "Current document chunks:",
            f"{int(supabase_status.get('document_chunks_current', 0)):,}",
        )
        st.write(
            "KPI observations:",
            f"{int(supabase_status.get('kpi_observations', 0)):,}",
        )
        st.write("Vector index:", "✅ HNSW")
    else:
        st.info("Storage: local reproducible fallback")
        st.write("KPI table:", "✅" if bootstrap and bootstrap.kpi_ready else "⚠️ docs-only")
        st.write("RAG sources:", "✅" if bootstrap and bootstrap.docs_ready else "❌")
        st.write("FAISS index:", "✅" if bootstrap and bootstrap.vector_ready else "❌")
        if supabase_runtime_configured() and not using_supabase:
            st.warning("Supabase configured but not seeded; using local storage.")
            with st.expander("Supabase status/error"):
                st.write(supabase_error or supabase_status)

    with st.expander("About this project"):
        st.markdown(
            """
            **Structured data:** real Ericsson/AERPAW 5G NSA KPI measurements.

            **RAG corpus:** 10 configured public sources spanning:
            - NR/LTE measurement standards,
            - NR data procedures and architecture,
            - the exact AERPAW Ericsson experiment,
            - Ericsson material on beamforming, coverage/capacity and network performance.

            **Retrieval:** BGE dense retrieval + lexical retrieval + reciprocal-rank fusion
            + cross-encoder reranking.

            **Storage:** Supabase/Postgres + pgvector in the hosted configuration, with
            local FAISS/BM25 retained as the notebook/development baseline.
            """
        )


observation = None

left, right = st.columns([1, 1])

with left:
    st.subheader("1. Select a measured KPI observation")

    if kpis is None or kpis.empty:
        st.info(
            "No KPI observations are available in the active backend, so the app is "
            "running in documentation-only RAG mode."
        )
    else:
        if "anomaly_score" in kpis.columns:
            default_df = kpis.sort_values(
                "anomaly_score", ascending=False, na_position="last"
            )
        else:
            default_df = kpis

        choices = default_df["observation_id"].astype(str).tolist()
        selected_id = st.selectbox("Observation", choices)
        row = kpis.loc[
            kpis["observation_id"].astype(str) == selected_id
        ].iloc[0]
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

    # Reserve units before the call. A failed provider request can still consume
    # resources, so the public-demo counter is conservative.
    st.session_state.request_units_used += units_needed

    try:
        retriever = cached_retriever(storage_backend)
        llm = cached_llm(provider, model)
        graph = build_graph(
            llm,
            retriever,
            reference_df=kpis,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
        )

        with st.spinner("Running the LangGraph workflow..."):
            result = graph.invoke(
                {
                    "question": cleaned_question,
                    "use_rag": use_rag,
                    "observation": observation,
                }
            )

        if provider == "cloudflare" and using_supabase:
            usage = result.get("llm_usage") or {}
            if usage.get("input_tokens", 0) or usage.get("output_tokens", 0):
                try:
                    record_cloudflare_usage(
                        model,
                        int(usage.get("input_tokens", 0)),
                        int(usage.get("output_tokens", 0)),
                    )
                    _render_cloudflare_quota(quota_placeholder, model)
                except Exception:
                    pass

        st.subheader("Answer")
        st.markdown(result["answer"])

        meta_cols = st.columns(6)
        meta_cols[0].metric("Route", result.get("route", "-"))
        meta_cols[1].metric("RAG", "On" if use_rag else "Off")
        meta_cols[2].metric("Storage", "Supabase" if using_supabase else "Local")
        meta_cols[3].metric(
            "Retrieval", result.get("retrieval_mode", retrieval_mode)
        )
        meta_cols[4].metric("Chunks", len(result.get("sources", [])))
        meta_cols[5].metric(
            "Latency", f"{result.get('latency_s', 0):.2f} s"
        )

        if result.get("retrieval_query"):
            with st.expander("Retrieval query"):
                st.code(result["retrieval_query"])

        if result.get("kpi_context"):
            with st.expander("Data-derived KPI context sent to the LLM"):
                st.code(result["kpi_context"])

        if result.get("sources"):
            st.subheader("Retrieved sources")
            for source in result["sources"]:
                page = (
                    f" — page {source['page']}"
                    if source.get("page")
                    else ""
                )
                section = (
                    f" — {source['section']}"
                    if source.get("section")
                    else ""
                )
                with st.expander(
                    f"[{source['citation']}] {source['source']}{page}{section}"
                ):
                    details = []
                    if source.get("retrieval_methods"):
                        details.append(
                            f"retrieved by {source['retrieval_methods']}"
                        )
                    if source.get("rerank_score") is not None:
                        details.append(
                            f"rerank score {source['rerank_score']:.3f}"
                        )
                    if details:
                        st.caption(" · ".join(details))
                    st.write(source["excerpt"])

        if compare and use_rag:
            baseline_graph = build_graph(
                llm,
                retriever,
                reference_df=kpis,
                top_k=top_k,
                retrieval_mode=retrieval_mode,
            )
            with st.spinner("Running the same LLM without retrieval..."):
                baseline = baseline_graph.invoke(
                    {
                        "question": cleaned_question,
                        "use_rag": False,
                        "observation": observation,
                    }
                )

            if provider == "cloudflare" and using_supabase:
                baseline_usage = baseline.get("llm_usage") or {}
                if (
                    baseline_usage.get("input_tokens", 0)
                    or baseline_usage.get("output_tokens", 0)
                ):
                    try:
                        record_cloudflare_usage(
                            model,
                            int(baseline_usage.get("input_tokens", 0)),
                            int(baseline_usage.get("output_tokens", 0)),
                        )
                        _render_cloudflare_quota(quota_placeholder, model)
                    except Exception:
                        pass

            st.subheader("Same LLM without RAG")
            st.markdown(baseline["answer"])
            st.caption(f"Latency: {baseline.get('latency_s', 0):.2f} s")

    except Exception as exc:
        st.error(
            "The request failed. Common causes are an unavailable local Ollama server, "
            "a Cloudflare Workers AI token/quota/capacity issue, missing Supabase "
            "migration/data, or a temporary provider error."
        )
        with st.expander("Technical error"):
            st.exception(exc)


st.divider()
st.caption(
    "KPI percentiles and anomaly scores are relative to this measurement dataset, not "
    "universal telecom quality thresholds. Retrieved sources are shown so the technical "
    "explanation can be inspected."
)
