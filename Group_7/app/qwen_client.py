import json
import os
import re
from typing import Dict, List, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv


RE_SCORE_KEYS = [
    "Relevance",
    "Empathy",
    "Specificity",
    "Credibility",
    "Objective",
    "Risk",
    "Engagement",
]

load_dotenv()


def _normalize_secret(value: str) -> str:
    normalized = (value or "").strip()
    # Handle common copy/paste quotes from docs or chat tools.
    return normalized.strip("'\"").strip()


def _is_ascii_text(value: str) -> bool:
    try:
        value.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _resolve_chat_completions_url(provider: str, api_base_url: str) -> str:
    provider_key = provider.lower().strip()
    base = (api_base_url or "").strip().rstrip("/")
    if provider_key == "deepseek":
        # Accept official DeepSeek-compatible forms:
        # - https://api.deepseek.com
        # - https://api.deepseek.com/v1
        # - full endpoint .../chat/completions
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"
    # DashScope compatible endpoint is already a concrete path.
    return base


def call_llm(
    messages: List[Dict[str, str]],
    temperature: float = 0.6,
    model: str = "qwen-plus",
    api_key: str = "",
    provider: str = "dashscope",
    api_base_url: str = "",
) -> str:
    provider_key = provider.lower().strip()
    if provider_key == "deepseek":
        default_key = os.getenv("DEEPSEEK_API_KEY", "")
        default_base = "https://api.deepseek.com/v1/chat/completions"
        default_model = "deepseek-chat"
    else:
        default_key = os.getenv("DASHSCOPE_API_KEY", "")
        default_base = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        default_model = "qwen-plus"

    key = _normalize_secret(api_key or default_key)
    final_model = (model or default_model).strip()
    final_url = _resolve_chat_completions_url(provider_key, (api_base_url or default_base).strip())

    if not key:
        if provider_key == "deepseek":
            return "DEEPSEEK_API_KEY is not set. Please set .env or input key in sidebar."
        return "DASHSCOPE_API_KEY is not set. Please set .env or input key in sidebar."
    if not _is_ascii_text(key):
        return (
            "API key contains non-ASCII characters. "
            "Please re-copy the key without Chinese quotes/spaces/newlines."
        )
    if not _is_ascii_text(final_url):
        return "API Base URL contains non-ASCII characters. Please use a standard ASCII URL."

    payload = {
        "model": final_model,
        "messages": messages,
        "temperature": temperature,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    try:
        resp = requests.post(final_url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as exc:  # pylint: disable=broad-except
        return f"LLM request failed ({provider_key}): {exc}"


def validate_llm_connection(
    provider: str,
    model: str,
    api_key: str,
    api_base_url: str,
) -> Tuple[bool, str]:
    test_messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Reply with OK only."},
    ]
    result = call_llm(
        messages=test_messages,
        temperature=0.0,
        model=model,
        api_key=api_key,
        provider=provider,
        api_base_url=api_base_url,
    )
    if result.startswith("LLM request failed") or result.endswith("is not set. Please set .env or input key in sidebar."):
        return False, result
    return True, f"Connected. Test response: {result[:80]}"


def build_personas(
    keyword: str,
    agg_row: pd.Series | None,
    corpus_df: pd.DataFrame,
    api_key: str = "",
    provider: str = "dashscope",
    model: str = "",
    api_base_url: str = "",
) -> str:
    metrics = agg_row.to_dict() if agg_row is not None else {}
    corpus = corpus_df["full_text"].head(3).tolist() if not corpus_df.empty else []
    prompt = [
        {
            "role": "system",
            "content": (
                "You are a product researcher. Generate 15 audience personas in Chinese. "
                "Output must be JSON array only, no markdown, no explanation, no code fences."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Topic: {keyword}\n"
                f"Comment aggregate metrics (JSON): {json.dumps(metrics, ensure_ascii=False)}\n"
                f"Language corpus samples: {json.dumps(corpus, ensure_ascii=False)}\n\n"
                "Return JSON array with 15 items, each containing: "
                "name, traits, concerns, speaking_style, likely_objection. "
                "Do NOT include any text before or after JSON."
            ),
        },
    ]
    return call_llm(
        prompt,
        temperature=0.7,
        api_key=api_key,
        provider=provider,
        model=model,
        api_base_url=api_base_url,
    )


def evaluate_copy_with_rescore(
    copy_text: str,
    persona_json: str,
    keyword: str,
    api_key: str = "",
    provider: str = "dashscope",
    model: str = "",
    api_base_url: str = "",
) -> str:
    prompt = [
        {
            "role": "system",
            "content": "You are a strict copywriting evaluator using RE-SCORE.",
        },
        {
            "role": "user",
            "content": (
                f"Topic: {keyword}\n"
                f"Persona: {persona_json}\n"
                f"Copy draft: {copy_text}\n\n"
                "Score each RE-SCORE dimension from 1-10 and explain briefly.\n"
                "Also generate one plausible Reddit-style comment from this persona.\n"
                "Return valid JSON:\n"
                "{"
                "\"scores\": {\"Relevance\":0,\"Empathy\":0,\"Specificity\":0,"
                "\"Credibility\":0,\"Objective\":0,\"Risk\":0,\"Engagement\":0},"
                "\"feedback\":\"...\","
                "\"simulated_comment\":\"...\""
                "}"
            ),
        },
    ]
    return call_llm(
        prompt,
        temperature=0.5,
        api_key=api_key,
        provider=provider,
        model=model,
        api_base_url=api_base_url,
    )


def improve_copy(
    original_copy: str,
    low_dimensions: List[str],
    content_gap_df: pd.DataFrame,
    eval_json: str,
    api_key: str = "",
    provider: str = "dashscope",
    model: str = "",
    api_base_url: str = "",
) -> str:
    fix = ""
    trust = ""
    if not content_gap_df.empty:
        fix = " | ".join(content_gap_df["suggested_content_fix"].dropna().head(3).tolist())
        trust = " | ".join(content_gap_df["missing_trust_signals"].dropna().head(3).tolist())
    prompt = [
        {"role": "system", "content": "You are a senior growth strategist and copywriter."},
        {
            "role": "user",
            "content": (
                f"Original copy:\n{original_copy}\n\n"
                f"Low score dimensions: {low_dimensions}\n"
                f"RE-SCORE evaluation JSON: {eval_json}\n"
                f"Suggested content fix from content_gap.csv: {fix}\n"
                f"Missing trust signals from content_gap.csv: {trust}\n\n"
                "Tasks:\n"
                "1) Rewrite the copy in English to improve low-score dimensions.\n"
                "2) Generate a detailed DALL-E 3 style image prompt in English.\n"
                "Return JSON: {\"rewritten_copy\":\"...\", \"image_prompt\":\"...\"}"
            ),
        },
    ]
    return call_llm(
        prompt,
        temperature=0.7,
        api_key=api_key,
        provider=provider,
        model=model,
        api_base_url=api_base_url,
    )


def try_parse_json(text: str) -> Dict:
    try:
        return json.loads(text)
    except Exception:
        raw = (text or "").strip()
        if not raw:
            return {}
        # Remove optional markdown code fences.
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
        # Try to extract JSON object/array wrapped by markdown or other text.
        for start, end in [("{", "}"), ("[", "]")]:
            si = raw.find(start)
            ei = raw.rfind(end)
            if si != -1 and ei != -1 and ei > si:
                candidate = raw[si : ei + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    continue
        return {}


def ensure_persona_json_array(
    raw_text: str,
    api_key: str,
    provider: str,
    model: str,
    api_base_url: str,
) -> str:
    parsed = try_parse_json(raw_text)
    if isinstance(parsed, list):
        return json.dumps(parsed, ensure_ascii=False)

    repair_prompt = [
        {
            "role": "system",
            "content": "Convert any text into valid JSON array only. Output JSON array only.",
        },
        {
            "role": "user",
            "content": (
                "The following content should represent persona list but is malformed. "
                "Repair it to valid JSON array. Keep at most 15 personas.\n\n"
                f"{raw_text}"
            ),
        },
    ]
    repaired = call_llm(
        repair_prompt,
        temperature=0.0,
        api_key=api_key,
        provider=provider,
        model=model,
        api_base_url=api_base_url,
    )
    repaired_parsed = try_parse_json(repaired)
    if isinstance(repaired_parsed, list):
        return json.dumps(repaired_parsed, ensure_ascii=False)
    return raw_text


def pick_low_dims(scores: Dict[str, float], threshold: float = 6.0) -> List[str]:
    lows = [k for k in RE_SCORE_KEYS if float(scores.get(k, 0)) < threshold]
    return lows or RE_SCORE_KEYS[:2]
