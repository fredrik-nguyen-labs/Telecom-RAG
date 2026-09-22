from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st

from telecom_rag.config import (
    CLOUDFLARE_GENERATOR_MODEL,
    CLOUDFLARE_ROUTER_MODEL,
    MAX_QUESTION_CHARS,
    MAX_REQUEST_UNITS_PER_SESSION,
    OLLAMA_MODEL,
    PROCESSED_KPI_PATH,
    TOP_K,
)
from telecom_rag.graph import build_graph
from telecom_rag.rag import get_llm
from telecom_rag.supabase_backend import (
    get_supabase_status,
    load_kpis_from_supabase,
    load_supabase_retriever,
    supabase_runtime_configured,
)


st.set_page_config(
    page_title="Telecom RAG",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
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
    "CLOUDFLARE_GENERATOR_MODEL",
    "CLOUDFLARE_ROUTER_MODEL",
    "CLOUDFLARE_EMBEDDING_MODEL",
    "CLOUDFLARE_RERANKER_MODEL",
    "USE_CLOUDFLARE_RETRIEVAL",
    "USE_SUPABASE",
    "SUPABASE_URL",
    "SUPABASE_PUBLISHABLE_KEY",
):
    secret_value = _read_secret(secret_name)
    if secret_value and not os.getenv(secret_name):
        os.environ[secret_name] = secret_value


@st.cache_resource(show_spinner="Preparing local assets...")
def cached_bootstrap():
    from telecom_rag.bootstrap import ensure_local_assets

    return ensure_local_assets()


@st.cache_resource(show_spinner="Loading retrieval backend...")
def cached_retriever(backend: str):
    if backend == "supabase":
        return load_supabase_retriever()

    from telecom_rag.rag import load_advanced_retriever

    return load_advanced_retriever(build_if_missing=False)


@st.cache_resource(show_spinner=False)
def cached_llm(provider: str, model: str, max_output_tokens: int | None = None):
    return get_llm(
        provider=provider,
        model=model,
        max_output_tokens=max_output_tokens,
    )


@st.cache_data(show_spinner=False)
def cached_kpis(backend: str) -> pd.DataFrame | None:
    if backend == "supabase":
        df = load_kpis_from_supabase()
        return None if df.empty else df

    if not PROCESSED_KPI_PATH.exists():
        return None

    from telecom_rag.data import load_processed_kpis

    return load_processed_kpis()


if "request_units_used" not in st.session_state:
    st.session_state.request_units_used = 0
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []


def _conversation_context(max_messages: int = 4, max_chars_per_message: int = 500) -> str:
    """Return a small recent-history window for follow-up questions."""
    lines: list[str] = []
    for message in st.session_state.chat_messages[-max_messages:]:
        role = "User" if message.get("role") == "user" else "Assistant"
        content = str(message.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content[:max_chars_per_message]}")
    return "\n".join(lines)


st.title("📡 5G Network Diagnostics RAG Assistant")
st.caption(
    "Grounded 5G diagnostics using real Ericsson/AERPAW measurements and "
    "telecom standards and technical documentation."
)
st.markdown(
    "🔗 [GitHub repository](https://github.com/fredrik-nguyen-labs/Telecom-RAG)"
)
st.caption(
    "Select a measurement or enter your own KPIs, ask a question, and review "
    "the data-backed diagnosis and cited evidence."
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
hosted_lightweight = os.getenv("HOSTED_LIGHTWEIGHT", "").lower() in {
    "1", "true", "yes", "on"
}

if using_supabase:
    bootstrap = None
    kpis = cached_kpis("supabase")
elif hosted_lightweight:
    st.error(
        "The lightweight hosted deployment requires a working, seeded Supabase backend."
    )
    with st.expander("Hosted backend details", expanded=True):
        if supabase_error:
            st.write("Supabase error:", supabase_error)
        else:
            st.write("Supabase status:", supabase_status)
        st.write(
            "Check USE_SUPABASE, SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY, "
            "and that the database has been seeded."
        )
    st.stop()
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
remaining_units = max(
    0, MAX_REQUEST_UNITS_PER_SESSION - st.session_state.request_units_used
)


def _optional_float(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _custom_observation_from_inputs(values: dict[str, str]) -> dict | None:
    observation: dict[str, object] = {
        "observation_id": "custom",
        "observation_source": "user-entered",
    }
    entered = False
    for key, raw in values.items():
        parsed = _optional_float(raw)
        if parsed is not None:
            observation[key] = parsed
            entered = True
    return observation if entered else None


def _citation_claims(answer: str) -> tuple[list[str], dict[str, list[str]]]:
    """Return citation IDs in first-use order and answer spans that cite each source."""
    order: list[str] = []
    claims: dict[str, list[str]] = {}

    spans = [
        span.strip()
        for span in re.split(r"(?<=[.!?])\s+|\n+", answer)
        if span.strip()
    ]
    for span in spans:
        citation_ids = [f"S{n}" for n in re.findall(r"\[S(\d+)\]", span)]
        if not citation_ids:
            continue
        clean_claim = re.sub(r"\s+", " ", span).strip()
        for citation_id in citation_ids:
            if citation_id not in order:
                order.append(citation_id)
            claims.setdefault(citation_id, [])
            if clean_claim not in claims[citation_id]:
                claims[citation_id].append(clean_claim)
    return order, claims


def _render_answer_cards(result: dict) -> None:
    """Render the model contract as distinct, easy-to-scan evidence cards."""
    sections = result.get("answer_sections") or {}
    if not sections:
        sections = {"answer": result.get("answer", "")}

    if sections.get("answer"):
        with st.container(border=True):
            st.markdown("### ✅ Answer")
            st.markdown(sections["answer"])

    secondary = []
    is_kpi_route = result.get("route") == "kpi+docs"

    observation_evidence = (
        sections.get("observation_evidence") or sections.get("measured_evidence")
    )
    if is_kpi_route and observation_evidence:
        secondary.append(("📊 Observation evidence", observation_evidence))
    if sections.get("technical_interpretation"):
        secondary.append(
            ("📚 Technical interpretation", sections["technical_interpretation"])
        )

    if len(secondary) == 2:
        cols = st.columns(2)
        for col, (title, body) in zip(cols, secondary):
            with col:
                with st.container(border=True):
                    st.markdown(f"### {title}")
                    st.markdown(body)
    else:
        for title, body in secondary:
            with st.container(border=True):
                st.markdown(f"### {title}")
                st.markdown(body)

    if is_kpi_route and sections.get("hypotheses"):
        with st.container(border=True):
            st.markdown("### 🧭 Hypotheses")
            st.markdown(sections["hypotheses"])
            st.caption(
                "Hypotheses are possible explanations, not measured facts. "
                "They are shown only for KPI-diagnostic questions."
            )


def _render_kpi_analytics(result: dict) -> None:
    """Render deterministic statistics produced before the LLM call."""
    if result.get("route") != "kpi+docs":
        return

    analysis = result.get("kpi_analysis") or {}
    if not analysis:
        return

    with st.container(border=True):
        st.markdown("### 📈 Statistical diagnostics")
        source_label = (
            "user-entered KPIs benchmarked against the Ericsson/AERPAW reference dataset"
            if analysis.get("source") == "user-entered"
            else "selected Ericsson/AERPAW measurement"
        )
        st.caption(
            f"Computed directly from the data layer for the {source_label}; "
            "these numbers are not generated by the LLM."
        )

        metrics = []
        neighbors = analysis.get("neighbors") or {}
        multivariate = analysis.get("multivariate") or {}
        anomaly = analysis.get("anomaly") or {}

        if multivariate.get("percentile") is not None:
            metrics.append(
                ("Multivariate rarity", f"{multivariate['percentile']:.0f}th pct")
            )
        if neighbors.get("throughput_median_mbps") is not None:
            metrics.append(
                (
                    "Similar-sample throughput",
                    f"{neighbors['throughput_median_mbps']:.1f} Mbps",
                )
            )
        if neighbors.get("throughput_gap_mbps") is not None:
            metrics.append(
                (
                    "Throughput vs neighbors",
                    f"{neighbors['throughput_gap_mbps']:+.1f} Mbps",
                )
            )
        if anomaly.get("percentile") is not None:
            metrics.append(
                ("Isolation Forest rarity", f"{anomaly['percentile']:.0f}th pct")
            )

        if metrics:
            cols = st.columns(len(metrics))
            for col, (label, value) in zip(cols, metrics):
                col.metric(label, value)

        percentiles = analysis.get("percentiles") or []
        if percentiles:
            st.markdown("**KPI percentiles**")
            pct_df = pd.DataFrame(
                [
                    {
                        "KPI": item["label"],
                        "Value": (
                            f"{item['value']:.3g} {item['unit']}".strip()
                        ),
                        "Dataset percentile": (
                            f"{item['percentile']:.0f}"
                            if item.get("percentile") is not None
                            else "—"
                        ),
                    }
                    for item in percentiles
                ]
            )
            st.dataframe(pct_df, hide_index=True, use_container_width=True)

        if neighbors:
            st.markdown("**Similar measured observations**")
            feature_names = ", ".join(neighbors.get("feature_labels") or [])
            if feature_names:
                st.write(
                    f"Matched the {neighbors.get('k', 0)} closest observations using "
                    f"{feature_names}. Throughput is excluded from the matching distance."
                )
            if neighbors.get("throughput_median_mbps") is not None:
                st.write(
                    "Neighbor throughput: "
                    f"median **{neighbors['throughput_median_mbps']:.1f} Mbps**, "
                    f"IQR **{neighbors['throughput_q25_mbps']:.1f}–"
                    f"{neighbors['throughput_q75_mbps']:.1f} Mbps**."
                )
            if neighbors.get("local_throughput_percentile") is not None:
                st.write(
                    "This observation's throughput is around the "
                    f"**{neighbors['local_throughput_percentile']:.0f}th percentile** "
                    "among those similar observations."
                )

        consistency = analysis.get("relationship_consistency") or []
        if consistency:
            st.markdown("**Cross-KPI consistency**")
            consistency_df = pd.DataFrame(
                [
                    {
                        "Target KPI": item["target_label"],
                        "Observed": round(item["actual"], 3),
                        "Similar-sample median": round(item["median"], 3),
                        "Local percentile": round(item["local_percentile"], 1),
                        "Assessment": item["status"],
                    }
                    for item in consistency
                ]
            )
            st.dataframe(
                consistency_df,
                hide_index=True,
                use_container_width=True,
            )

        key_relationships = analysis.get("key_relationships") or []
        if key_relationships:
            st.markdown("**Key KPI relationships in the reference dataset**")
            relation_df = pd.DataFrame(
                [
                    {
                        "Relationship": f"{item['x_label']} ↔ {item['y_label']}",
                        "Rank correlation (ρ)": round(item["spearman_rho"], 3),
                        "Observations": item["n"],
                    }
                    for item in key_relationships
                ]
            )
            st.dataframe(
                relation_df,
                hide_index=True,
                use_container_width=True,
            )

        correlations = analysis.get("throughput_correlations") or []
        if correlations:
            with st.expander("More throughput correlations"):
                corr_df = pd.DataFrame(
                    [
                        {
                            "KPI": item["label"],
                            "Rank correlation (ρ)": round(item["spearman_rho"], 3),
                            "Observations": item["n"],
                        }
                        for item in correlations
                    ]
                )
                st.dataframe(corr_df, hide_index=True, use_container_width=True)

        if key_relationships or correlations:
            st.caption(
                "Rank correlations are descriptive associations in this dataset; "
                "they do not establish causality."
            )


def _render_cited_evidence(answer: str, sources: list[dict]) -> None:
    """Show only sources actually cited by the answer, with claim and exact chunk."""
    if not sources:
        return

    citation_order, claims = _citation_claims(answer)
    if not citation_order:
        st.info(
            "The answer did not include inline [S#] markers. The retrieved evidence "
            "used for this response is still available below."
        )
        st.subheader("Retrieved evidence")
        for rank, source in enumerate(sources[:3], start=1):
            title = source.get("title") or source.get("source") or "Unknown source"
            with st.expander(f"{rank}. {title}", expanded=rank == 1):
                st.code(
                    source.get("content") or source.get("excerpt") or "",
                    language=None,
                )
        return

    by_id = {source.get("citation"): source for source in sources}
    st.subheader("Cited evidence")
    st.caption(
        "Each card links the claim in the answer to the exact retrieved chunk supplied "
        "to the model. Page/section metadata comes from the source document."
    )

    for rank, citation_id in enumerate(citation_order, start=1):
        source = by_id.get(citation_id)
        if not source:
            st.warning(f"[{citation_id}] was cited in the answer but is not in the retrieved sources.")
            continue

        location_bits = []
        if source.get("page"):
            location_bits.append(f"page {source['page']}")
        if source.get("section"):
            location_bits.append(f"section {source['section']}")
        if source.get("chunk_id"):
            location_bits.append(f"chunk {source['chunk_id']}")
        location = " · ".join(location_bits) or "location metadata unavailable"

        title = source.get("title") or source.get("source") or "Unknown source"
        label = f"{rank}. [{citation_id}] {title} — {location}"

        with st.expander(label, expanded=rank <= 2):
            st.markdown("**Claim(s) in the answer using this citation**")
            for claim in claims.get(citation_id, []):
                st.markdown(f"- {claim}")

            st.markdown("**Exact retrieved evidence chunk**")
            st.code(source.get("content") or source.get("excerpt") or "", language=None)


if cloudflare_available:
    provider = "cloudflare"
    model = os.getenv("CLOUDFLARE_GENERATOR_MODEL", CLOUDFLARE_GENERATOR_MODEL)
    router_model = os.getenv("CLOUDFLARE_ROUTER_MODEL", CLOUDFLARE_ROUTER_MODEL)
elif hosted_lightweight:
    st.error("Hosted inference is not configured.")
    st.stop()
else:
    provider = "ollama"
    model = os.getenv("OLLAMA_MODEL", OLLAMA_MODEL)
    router_model = model

retrieval_mode = "reranked"
top_k = TOP_K


observation = None

left, right = st.columns([1, 1])

with left:
    st.subheader("1. Provide KPI context")

    observation_mode = st.radio(
        "Observation source",
        ["Dataset sample", "Enter your own KPIs"],
        horizontal=True,
        help=(
            "Use a real Ericsson/AERPAW row for a reproducible example, or enter your own "
            "radio KPIs and compare them with the dataset distribution."
        ),
    )

    if observation_mode == "Dataset sample":
        if kpis is None or kpis.empty:
            st.info(
                "No KPI observations are available in the active backend. "
                "Use 'Enter your own KPIs' or ask a documentation-only question."
            )
        else:
            if "anomaly_score" in kpis.columns:
                default_df = kpis.sort_values(
                    "anomaly_score", ascending=False, na_position="last"
                )
            else:
                default_df = kpis

            choices = default_df["observation_id"].astype(str).tolist()
            selected_id = st.selectbox("Dataset observation", choices)
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
            st.caption(
                "A selected row is context only. Generic questions such as "
                "'What does RSRP stand for?' still use the docs-only route."
            )
    else:
        st.caption(
            "Enter only the KPIs you know. Blank fields are ignored. Values are compared "
            "with the real Ericsson/AERPAW dataset when KPI analysis is used."
        )

        custom_left, custom_right = st.columns(2)
        with custom_left:
            nr_rsrp = st.text_input("NR RSRP (dBm)", placeholder="-95")
            nr_sinr = st.text_input("NR SINR (dB)", placeholder="10")
            nr_cqi = st.text_input("NR CQI", placeholder="12")
            nr_ri = st.text_input("NR RI", placeholder="2")
        with custom_right:
            lte_rsrp = st.text_input("LTE RSRP (dBm)", placeholder="-90")
            lte_sinr = st.text_input("LTE SINR (dB)", placeholder="15")
            nr_mcs = st.text_input("NR MCS", placeholder="18")
            throughput = st.text_input("Throughput (Mbps)", placeholder="50")

        custom_values = {
            "nr_rsrp_dbm": nr_rsrp,
            "nr_sinr_db": nr_sinr,
            "nr_cqi": nr_cqi,
            "nr_ri": nr_ri,
            "lte_rsrp_dbm": lte_rsrp,
            "lte_sinr_db": lte_sinr,
            "nr_mcs": nr_mcs,
            "throughput_mbps": throughput,
        }
        observation = _custom_observation_from_inputs(custom_values)

        if observation:
            preview_rows = [
                {"metric": key, "value": value}
                for key, value in observation.items()
                if key not in {"observation_id", "observation_source"}
            ]
            st.dataframe(
                pd.DataFrame(preview_rows),
                hide_index=True,
                use_container_width=True,
            )
            st.success(
                "Custom KPI observation ready. Ask about 'these values', "
                "'my measurement', or 'this observation' to activate KPI analysis."
            )
        else:
            st.info(
                "Enter at least one KPI value to create a custom observation."
            )


with right:
    st.subheader("2. Ask a question")

    if st.session_state.chat_messages:
        if st.button("New conversation"):
            st.session_state.chat_messages = []
            st.rerun()

        for message in st.session_state.chat_messages[-8:]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if observation_mode == "Enter your own KPIs" and observation:
        default_q = (
            "Diagnose these values. What stands out, what could explain the throughput, "
            "and what should I investigate next?"
        )
    elif observation:
        default_q = (
            "Why might this observation have this throughput, and which radio measurements "
            "are most relevant to investigate?"
        )
    else:
        default_q = "What do RSRP and SINR measure in a 5G NR network?"

    if st.session_state.chat_messages:
        default_q = ""

    question = st.text_area(
        "Question",
        value=default_q,
        height=130,
        max_chars=MAX_QUESTION_CHARS,
        help=f"Maximum {MAX_QUESTION_CHARS} characters per request.",
    )

    units_needed = 1
    run_disabled = units_needed > remaining_units

    if run_disabled:
        st.warning("This session has reached its request limit.")

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
    # resources, so the session counter is conservative.
    st.session_state.request_units_used += units_needed

    try:
        retriever = cached_retriever(storage_backend)
        llm = cached_llm(provider, model)
        router_llm = (
            cached_llm(
                "cloudflare",
                os.getenv("CLOUDFLARE_ROUTER_MODEL", CLOUDFLARE_ROUTER_MODEL),
                8,
            )
            if provider == "cloudflare"
            else llm
        )
        graph = build_graph(
            llm,
            retriever,
            router_llm=router_llm,
            reference_df=kpis,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
        )

        conversation_context = _conversation_context()
        with st.spinner("Running the LangGraph workflow..."):
            result = graph.invoke(
                {
                    "question": cleaned_question,
                    "conversation_context": conversation_context,
                    "use_rag": True,
                    "observation": observation,
                }
            )

        answer_text = str(result.get("answer", "")).strip()
        if not answer_text:
            raise RuntimeError("The model returned an empty visible answer.")

        st.session_state.chat_messages.extend(
            [
                {"role": "user", "content": cleaned_question},
                {"role": "assistant", "content": answer_text},
            ]
        )

        _render_answer_cards(result)
        _render_kpi_analytics(result)

        sources = result.get("sources", [])
        _render_cited_evidence(answer_text, sources)

        with st.expander("Technical details"):
            st.write(
                "Route:",
                "KPI + docs" if result.get("route") == "kpi+docs" else "Docs only",
            )
            st.write("Retrieval:", result.get("retrieval_mode", retrieval_mode))
            st.write("Retrieved chunks:", len(sources))
            st.write(
                "Latency:",
                {
                    "routing_s": round(float(result.get("router_latency_s", 0)), 2),
                    "retrieval_s": round(float(result.get("retrieval_latency_s", 0)), 2),
                    "generation_s": round(
                        float(result.get("generation_latency_s", result.get("latency_s", 0))),
                        2,
                    ),
                },
            )

    except Exception as exc:
        st.error(
            "The request failed. Check the technical error below for a Cloudflare, "
            "Supabase, retrieval, or model-response issue."
        )
        with st.expander("Technical error"):
            st.exception(exc)


st.divider()
st.caption(
    "KPI percentiles, correlations, nearest-neighbor comparisons and anomaly/rarity "
    "statistics are relative to this measurement dataset, not universal telecom quality "
    "thresholds. Retrieved sources are shown so the technical explanation can be inspected."
)

st.markdown("#### Tech stack")
st.caption(
    "Python · Streamlit · LangGraph · LangChain · Supabase/PostgreSQL + pgvector/FTS · "
    "Cloudflare Workers AI · BGE embeddings/reranking"
)

