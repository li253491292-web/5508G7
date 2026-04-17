import os


def resolve_llm_config() -> tuple[str, str, str, str]:
    provider = os.getenv("LLM_PROVIDER", "dashscope").strip().lower()

    if provider == "deepseek":
        api_key = os.getenv("LLM_API_KEY", "").strip() or os.getenv("DEEPSEEK_API_KEY", "").strip()
        base = os.getenv("LLM_BASE_URL", "").strip() or "https://api.deepseek.com"
        model = os.getenv("LLM_MODEL", "").strip() or "deepseek-chat"
        base = base.rstrip("/")
        if base.endswith("/chat/completions"):
            url = base
        elif base.endswith("/v1"):
            url = f"{base}/chat/completions"
        else:
            url = f"{base}/v1/chat/completions"
        return provider, api_key, url, model

    # default: DashScope compatible endpoint
    provider = "dashscope"
    api_key = os.getenv("LLM_API_KEY", "").strip() or os.getenv("DASHSCOPE_API_KEY", "").strip()
    url = os.getenv("LLM_BASE_URL", "").strip() or "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    model = os.getenv("LLM_MODEL", "").strip() or "qwen-plus"
    return provider, api_key, url, model
