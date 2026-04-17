import argparse
import getpass
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CRAWLER_DIR = PROJECT_ROOT / "data_crawler"
DATA_SOURCE_DIR = PROJECT_ROOT / "data_sourse"
CLEANED_DIR = PROJECT_ROOT.parent / "5506" / "cleaned"

FEATURES_DIR = CLEANED_DIR / "features"
AGG_FILE = CLEANED_DIR / "aggregation" / "post_comment_aggregate.csv"
GAP_FILE = CLEANED_DIR / "content_gap" / "content_gap.csv"


def _run_script(script_name: str, env: dict) -> None:
    script_path = DATA_CRAWLER_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    print(f"\n>>> Running {script_name}")
    subprocess.run([sys.executable, str(script_path)], cwd=str(DATA_CRAWLER_DIR), env=env, check=True)


def _merge_feature_files() -> Path:
    files = sorted(FEATURES_DIR.glob("post_feature_*.csv"))
    if not files:
        raise FileNotFoundError(f"No feature files found under {FEATURES_DIR}")
    dfs = [pd.read_csv(f, low_memory=False) for f in files]
    merged = pd.concat(dfs, ignore_index=True)
    DATA_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    output = DATA_SOURCE_DIR / "all_features.csv"
    merged.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Merged {len(files)} feature files -> {output} ({len(merged)} rows)")
    return output


def _sync_outputs_to_app() -> None:
    DATA_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    _merge_feature_files()

    if not AGG_FILE.exists():
        raise FileNotFoundError(f"Missing aggregate file: {AGG_FILE}")
    shutil.copy2(AGG_FILE, DATA_SOURCE_DIR / "post_comment_aggregate.csv")
    print(f"Copied aggregate -> {DATA_SOURCE_DIR / 'post_comment_aggregate.csv'}")

    if not GAP_FILE.exists():
        raise FileNotFoundError(f"Missing content gap file: {GAP_FILE}")
    shutil.copy2(GAP_FILE, DATA_SOURCE_DIR / "content_gap.csv")
    print(f"Copied content gap -> {DATA_SOURCE_DIR / 'content_gap.csv'}")


def _build_vector(env: dict, max_docs: int) -> None:
    print("\n>>> Building vector store for app")
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "data_pipeline.py"), "--force", "--max-docs", str(max_docs)],
        cwd=str(PROJECT_ROOT),
        env=env,
        check=True,
    )


def _prepare_env(args: argparse.Namespace) -> dict:
    env = os.environ.copy()
    provider = args.provider.strip().lower()
    env["LLM_PROVIDER"] = provider

    api_key = (args.api_key or "").strip()
    if not api_key:
        api_key = getpass.getpass(f"Input {provider} API key (hidden): ").strip()
    if not api_key:
        raise ValueError("API key is required for LLM stages.")

    env["LLM_API_KEY"] = api_key
    if provider == "deepseek":
        env["DEEPSEEK_API_KEY"] = api_key
        env["LLM_BASE_URL"] = (args.api_base_url or "https://api.deepseek.com").strip()
        env["LLM_MODEL"] = (args.model or "deepseek-chat").strip()
    else:
        env["DASHSCOPE_API_KEY"] = api_key
        env["LLM_BASE_URL"] = (args.api_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions").strip()
        env["LLM_MODEL"] = (args.model or "qwen-plus").strip()
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description="One-click mining + processing + vector pipeline")
    parser.add_argument("--provider", choices=["dashscope", "deepseek"], default="dashscope")
    parser.add_argument("--api-key", default="", help="Optional API key; if empty will prompt securely.")
    parser.add_argument("--api-base-url", default="", help="Optional base URL override.")
    parser.add_argument("--model", default="", help="Optional model override.")
    parser.add_argument("--max-docs", type=int, default=50000, help="Max docs for vector rebuild.")
    parser.add_argument("--skip-vector", action="store_true", help="Skip vector rebuild stage.")
    args = parser.parse_args()

    env = _prepare_env(args)

    # Ordered stages (from processed files to app-ready outputs).
    _run_script("post_feature.py", env)
    _run_script("comments_labels.py", env)
    _run_script("post_comment_aggregate.py", env)
    _run_script("content_gap.py", env)

    _sync_outputs_to_app()
    if not args.skip_vector:
        _build_vector(env, max_docs=args.max_docs)

    print("\n✅ Pipeline completed. UI can now run without building RAG inside app.")


if __name__ == "__main__":
    main()
