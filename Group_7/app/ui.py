import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.data_engine import data_files_exist, get_full_data
from app.persona_store import get_persona, upsert_persona
from app.qwen_client import (
    RE_SCORE_KEYS,
    build_personas,
    ensure_persona_json_array,
    evaluate_copy_with_rescore,
    improve_copy,
    pick_low_dims,
    try_parse_json,
    validate_llm_connection,
)
from app.rag_engine import (
    find_chroma_sqlite_files,
    find_existing_vector_store,
    retrieve_corpus,
)


def make_radar(scores: dict[str, float]) -> go.Figure:
    values = [float(scores.get(k, 0)) for k in RE_SCORE_KEYS]
    labels = RE_SCORE_KEYS + [RE_SCORE_KEYS[0]]
    vals = values + [values[0]]
    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=vals,
                theta=labels,
                fill="toself",
                name="RE-SCORE",
            )
        ]
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=False,
        margin=dict(l=30, r=30, t=30, b=30),
    )
    return fig


def _infer_keyword_from_text(text: str, keywords: list[str]) -> str:
    if not keywords:
        return ""
    content = (text or "").strip().lower()
    if not content:
        return keywords[0]

    # Strong signal: keyword appears directly in content.
    for kw in keywords:
        if kw and kw.lower() in content:
            return kw

    # Weak signal: token overlap score.
    tokens = set(t for t in re.split(r"[^a-z0-9_]+", content) if t)
    if not tokens:
        return keywords[0]
    best_kw = keywords[0]
    best_score = -1
    for kw in keywords:
        kw_tokens = set(t for t in re.split(r"[^a-z0-9_]+", kw.lower()) if t)
        score = len(tokens.intersection(kw_tokens))
        if score > best_score:
            best_score = score
            best_kw = kw
    return best_kw


def _extract_keywords_from_full_data(full_data: dict[str, pd.DataFrame]) -> list[str]:
    candidates: list[str] = []
    for table_name in ["all_features", "post_comment_aggregate", "content_gap"]:
        df = full_data.get(table_name, pd.DataFrame())
        if "source_keyword" not in df.columns or df.empty:
            continue
        vals = (
            df["source_keyword"]
            .astype(str)
            .str.strip()
            .replace("nan", "")
            .replace("None", "")
        )
        vals = vals[vals != ""]
        candidates.extend(vals.tolist())
    if not candidates:
        return []
    uniq = sorted(set(candidates))
    return uniq


def _parse_persona_list(persona_text: str) -> list[dict]:
    parsed = try_parse_json(persona_text)
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, dict):
        # Be tolerant to wrapped outputs like {"personas":[...]}.
        for key in ["personas", "data", "items", "results"]:
            value = parsed.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _aggregate_comment_metrics(agg_df: pd.DataFrame) -> pd.Series | None:
    if agg_df.empty:
        return None
    numeric_df = agg_df.apply(pd.to_numeric, errors="coerce")
    means = numeric_df.mean(numeric_only=True).dropna()
    if means.empty:
        return None
    return means


def _score_one_persona(
    persona_item: dict,
    idx: int,
    copy_demo: str,
    selected_keyword: str,
    api_key: str,
    provider_key: str,
    model_name: str,
    api_base_url: str,
) -> dict:
    persona_name = str(persona_item.get("name", f"Persona-{idx}"))
    eval_text = evaluate_copy_with_rescore(
        copy_demo,
        json.dumps(persona_item, ensure_ascii=False),
        selected_keyword,
        api_key=api_key,
        provider=provider_key,
        model=model_name,
        api_base_url=api_base_url,
    )
    parsed_eval = try_parse_json(eval_text)
    scores = parsed_eval.get("scores", {}) if isinstance(parsed_eval, dict) else {}
    row = {
        "persona_name": persona_name,
        "feedback": parsed_eval.get("feedback", "") if isinstance(parsed_eval, dict) else "",
        "simulated_comment": parsed_eval.get("simulated_comment", "") if isinstance(parsed_eval, dict) else "",
    }
    for dim in RE_SCORE_KEYS:
        row[dim] = float(scores.get(dim, 0)) if isinstance(scores, dict) else 0.0
    return row


@st.cache_data(ttl=60, show_spinner=False)
def _check_llm_connection_cached(
    provider: str,
    model: str,
    api_key: str,
    api_base_url: str,
) -> tuple[bool, str]:
    return validate_llm_connection(
        provider=provider,
        model=model,
        api_key=api_key,
        api_base_url=api_base_url,
    )


def run_app() -> None:
    st.set_page_config(page_title="RAG Content Simulation Engine", layout="wide")
    st.title("RAG Content Simulation & Optimization Engine")

    if not data_files_exist():
        st.error("CSV files not found in data_sourse folder.")
        return

    full_data = get_full_data()
    keywords = _extract_keywords_from_full_data(full_data)

    copy_demo = st.text_area("Input copy demo", height=150, placeholder="Enter your demo copy...")
    selected_keyword = _infer_keyword_from_text(copy_demo, keywords) if keywords else ""
    if not selected_keyword:
        selected_keyword = (copy_demo.strip()[:64] if copy_demo.strip() else "general_topic")

    with st.sidebar:
        st.header("Data & Index")
        st.markdown("### LLM API Configuration")
        llm_provider = st.selectbox("LLM provider", ["DashScope (Qwen)", "DeepSeek"], index=0)
        is_deepseek = llm_provider.startswith("DeepSeek")
        env_key_name = "DEEPSEEK_API_KEY" if is_deepseek else "DASHSCOPE_API_KEY"
        default_key = os.getenv(env_key_name, "")
        default_model = "deepseek-chat" if is_deepseek else "qwen-plus"
        default_base_url = (
            "https://api.deepseek.com"
            if is_deepseek
            else "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        )

        model_name = st.text_input("Model", value=default_model)
        api_base_url = st.text_input("API Base URL", value=default_base_url)
        api_mode = st.radio(
            "API key source",
            options=["Use .env / environment", "Input in app (temporary)"],
            index=0,
        )
        manual_api_key = st.text_input(
            f"{'DeepSeek' if is_deepseek else 'DashScope'} API Key",
            type="password",
            placeholder="sk-...",
            help="Only kept in current Streamlit session.",
        )
        api_key = manual_api_key.strip() if api_mode == "Input in app (temporary)" else default_key.strip()
        if api_key:
            st.success("API key configured.")
            provider_key = "deepseek" if is_deepseek else "dashscope"
            ok_conn, conn_msg = _check_llm_connection_cached(
                provider=provider_key,
                model=model_name,
                api_key=api_key,
                api_base_url=api_base_url,
            )
            if ok_conn:
                st.success("LLM connection available.")
            else:
                st.error(f"LLM connection failed: {conn_msg}")
        else:
            st.warning("No API key configured yet. LLM features will fail until you set one.")

        detected_chroma = find_chroma_sqlite_files()
        detected_faiss = find_existing_vector_store()
        rag_backend = "Chroma(SQLite)" if detected_chroma else "FAISS"

        if detected_faiss:
            st.success(f"Detected FAISS files in: `{detected_faiss[0].parent}`")
        if detected_chroma:
            st.success(f"Detected Chroma sqlite files: {len(detected_chroma)} (will be used together)")
        else:
            st.info("No Chroma sqlite files detected yet.")
        if not detected_faiss:
            st.info("No FAISS + parquet vector files detected yet.")
        st.caption(f"Keyword matching mode: auto (current: {selected_keyword or 'N/A'})")
        st.caption(f"RAG backend mode: auto (current: {rag_backend})")
        st.caption("Vector store is managed by offline pipeline: `python crawler/run_pipeline.py`.")
        st.caption("Set env var API key or input in sidebar before using LLM features.")

    agg_df = full_data["post_comment_aggregate"]
    gap_df = full_data["content_gap"]
    backend_key = "chroma" if rag_backend.startswith("Chroma") else "faiss"
    retrieval_query = selected_keyword if selected_keyword else (copy_demo.strip() or "general_topic")
    corpus_df = retrieve_corpus(
        retrieval_query,
        top_k=10,
        backend=backend_key,
        chroma_sqlite_path="",
    )

    st.subheader("Phase 1 - Data Engine")
    st.markdown(f"**Auto matched keyword:** `{selected_keyword}`")
    c1, c2, c3 = st.columns(3)
    c1.metric("all_features rows", len(full_data["all_features"]))
    c2.metric("post_comment_aggregate rows", len(agg_df))
    c3.metric("content_gap rows", len(gap_df))
    with st.expander("Preview topic data"):
        st.dataframe(full_data["all_features"].head(10))
        st.dataframe(agg_df.head(10))
        st.dataframe(gap_df.head(10))

    st.subheader("Phase 2 - Persona Hub")
    if not corpus_df.empty:
        st.write("Top 10 retrieved language corpus rows (from full_text vector search)")
        st.dataframe(corpus_df[["post_id", "source_keyword", "subreddit", "similarity", "full_text"]].head(10))
    else:
        st.info("No vector index loaded yet. Please initialize vector DB first.")

    agg_row = _aggregate_comment_metrics(agg_df)
    cached_persona = get_persona(selected_keyword)
    # Backward compatibility: repair old cached persona formats.
    if cached_persona:
        cached_list = _parse_persona_list(cached_persona)
        if cached_list:
            cached_persona = json.dumps(cached_list, ensure_ascii=False)
            upsert_persona(selected_keyword, cached_persona)

    if cached_persona and st.session_state.get("persona_keyword") != selected_keyword:
        st.session_state["persona_text"] = cached_persona
        st.session_state["persona_keyword"] = selected_keyword

    col_a, col_b = st.columns(2)
    if col_a.button("Generate / Refresh Persona with LLM"):
        provider_key = "deepseek" if is_deepseek else "dashscope"
        persona_raw = build_personas(
            selected_keyword,
            agg_row,
            corpus_df,
            api_key=api_key,
            provider=provider_key,
            model=model_name,
            api_base_url=api_base_url,
        )
        persona_text = ensure_persona_json_array(
            raw_text=persona_raw,
            api_key=api_key,
            provider=provider_key,
            model=model_name,
            api_base_url=api_base_url,
        )
        st.session_state["persona_text"] = persona_text
        st.session_state["persona_keyword"] = selected_keyword
        upsert_persona(selected_keyword, persona_text)
    if col_b.button("Load Cached Persona"):
        if cached_persona:
            st.session_state["persona_text"] = cached_persona
            st.session_state["persona_keyword"] = selected_keyword
        else:
            st.info("No cached persona found for this keyword yet.")

    if "persona_text" in st.session_state:
        if cached_persona:
            st.caption("Persona cache: found and reusable for this keyword.")
        else:
            st.caption("Persona cache: none yet for this keyword.")
        persona_list = _parse_persona_list(st.session_state["persona_text"])
        st.markdown(f"**Persona Output ({len(persona_list)} personas)**")
        if persona_list:
            st.dataframe(pd.DataFrame(persona_list), use_container_width=True)
        else:
            st.warning("Persona output is not valid JSON array yet.")

    st.subheader("Phase 3 - Simulation Lab (RE-SCORE)")
    if st.button("Run Persona Simulation & Score"):
        if not copy_demo.strip():
            st.warning("Please input copy text first.")
        else:
            persona_list = _parse_persona_list(st.session_state.get("persona_text", ""))
            if not persona_list:
                st.warning("Please generate personas first.")
            else:
                provider_key = "deepseek" if is_deepseek else "dashscope"
                eval_rows: list[dict] = []
                total = len(persona_list)
                done = 0
                progress = st.progress(0, text="Scoring personas in parallel...")
                max_workers = min(6, max(1, total))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_idx = {
                        executor.submit(
                            _score_one_persona,
                            persona_item,
                            idx,
                            copy_demo,
                            selected_keyword,
                            api_key,
                            provider_key,
                            model_name,
                            api_base_url,
                        ): idx
                        for idx, persona_item in enumerate(persona_list, start=1)
                    }
                    for future in as_completed(future_to_idx):
                        try:
                            eval_rows.append(future.result())
                        except Exception as exc:  # pylint: disable=broad-except
                            idx = future_to_idx[future]
                            fallback = {"persona_name": f"Persona-{idx}", "feedback": f"Evaluation failed: {exc}", "simulated_comment": ""}
                            for dim in RE_SCORE_KEYS:
                                fallback[dim] = 0.0
                            eval_rows.append(fallback)
                        done += 1
                        progress.progress(done / total, text=f"Scoring personas in parallel... ({done}/{total})")
                progress.empty()

                eval_df = pd.DataFrame(eval_rows)
                st.session_state["eval_table"] = eval_df
                if not eval_df.empty:
                    avg_scores = {dim: float(eval_df[dim].mean()) for dim in RE_SCORE_KEYS}
                    st.session_state["avg_scores"] = avg_scores
                else:
                    st.session_state["avg_scores"] = {}

    if "eval_table" in st.session_state:
        eval_df = st.session_state["eval_table"]
        st.markdown("**Persona Simulation Table**")
        st.dataframe(eval_df, use_container_width=True)
        avg_scores = st.session_state.get("avg_scores", {})
        if avg_scores:
            st.markdown("**Average RE-SCORE (all personas)**")
            avg_df = pd.DataFrame([avg_scores])
            st.dataframe(avg_df, use_container_width=True)
            st.plotly_chart(make_radar(avg_scores), use_container_width=True)

    st.subheader("Phase 4 - Optimization Engine")
    if st.button("Rewrite Copy + Generate Image Prompt"):
        eval_df = st.session_state.get("eval_table", pd.DataFrame())
        avg_scores = st.session_state.get("avg_scores", {})
        if not copy_demo.strip() or eval_df.empty or not avg_scores:
            st.warning("Run scoring first and ensure simulation table is available.")
        else:
            low_dims = pick_low_dims(avg_scores)
            provider_key = "deepseek" if is_deepseek else "dashscope"
            improved = improve_copy(
                copy_demo,
                low_dims,
                gap_df,
                json.dumps(avg_scores, ensure_ascii=False),
                api_key=api_key,
                provider=provider_key,
                model=model_name,
                api_base_url=api_base_url,
            )
            st.session_state["improved_text"] = improved
            st.session_state["improved_json"] = try_parse_json(improved)

    if "improved_text" in st.session_state:
        st.markdown("**Optimization Output**")
        improved_json = st.session_state.get("improved_json", {})
        rewritten = improved_json.get("rewritten_copy", "")
        image_prompt = improved_json.get("image_prompt", "")
        summary_table = pd.DataFrame(
            [
                {
                    "rewritten_copy_preview": (rewritten[:180] + "...") if len(rewritten) > 180 else rewritten,
                    "image_prompt_preview": (image_prompt[:180] + "...") if len(image_prompt) > 180 else image_prompt,
                }
            ]
        )
        st.dataframe(summary_table, use_container_width=True)
        if rewritten:
            st.markdown("**Rewritten copy**")
            st.write(rewritten)
        if image_prompt:
            st.markdown("**Image prompt (DALL-E 3 style)**")
            st.code(image_prompt)
            image_url = f"https://image.pollinations.ai/prompt/{quote(image_prompt)}"
            st.image(image_url, caption="Pollinations preview", use_container_width=True)
