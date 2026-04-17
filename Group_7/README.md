# RAG Content Simulation & Optimization Engine

## Structure

- `app/`: Streamlit application modules and entry (`app/main.py`)
- `crawler/`: one-click pipeline entry (`crawler/run_pipeline.py`)
- `crawler/scripts/`: archived crawler/data-processing implementations
- `data_crawler/`: compatibility wrappers for old script paths
- `data_sourse/`: app input/output datasets
- `vector_db/`: generated FAISS index artifacts

## Quick Start

1) Install dependencies

```bash
pip install -r requirements.txt
```

2) Configure API keys (optional for retrieval-only mode)

```bash
copy .env.example .env
```

3) Run one-click mining + processing + vector build

```bash
python crawler/run_pipeline.py --provider deepseek
```

or

```bash
python crawler/run_pipeline.py --provider dashscope
```

- If `--api-key` is omitted, key is requested securely in terminal.
- Supports both DeepSeek and DashScope(Qwen).
- Produces app-ready CSV files in `data_sourse/` and rebuilds vector store.

4) Launch app

```bash
streamlit run app/main.py
```

Backward-compatible entry still works:

```bash
streamlit run app.py
```

## Required Outputs

After pipeline, these files should exist in `data_sourse/`:

- `all_features.csv`
- `post_comment_aggregate.csv`
- `content_gap.csv`

## Notes

- UI no longer includes RAG build buttons; vector build is managed offline by pipeline.
- Retrieval supports FAISS and auto-detected Chroma SQLite.
