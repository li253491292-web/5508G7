from pathlib import Path
from typing import Dict

import duckdb
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data_sourse"
ALL_FEATURES_CSV = DATA_DIR / "all_features.csv"
POST_COMMENT_CSV = DATA_DIR / "post_comment_aggregate.csv"
CONTENT_GAP_CSV = DATA_DIR / "content_gap.csv"


def data_files_exist() -> bool:
    return ALL_FEATURES_CSV.exists() and POST_COMMENT_CSV.exists() and CONTENT_GAP_CSV.exists()


@st.cache_resource
def get_duckdb_conn() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(database=":memory:")
    conn.execute(
        f"""
        CREATE VIEW all_features AS
        SELECT * FROM read_csv_auto(
            '{ALL_FEATURES_CSV.as_posix()}',
            ignore_errors=true,
            sample_size=-1,
            all_varchar=true
        );
        """
    )
    conn.execute(
        f"""
        CREATE VIEW post_comment_aggregate AS
        SELECT * FROM read_csv_auto(
            '{POST_COMMENT_CSV.as_posix()}',
            ignore_errors=true,
            sample_size=-1,
            all_varchar=true
        );
        """
    )
    conn.execute(
        f"""
        CREATE VIEW content_gap AS
        SELECT * FROM read_csv_auto(
            '{CONTENT_GAP_CSV.as_posix()}',
            ignore_errors=true,
            sample_size=-1,
            all_varchar=true
        );
        """
    )
    return conn


def get_keywords() -> list[str]:
    conn = get_duckdb_conn()
    candidates = [
        "SELECT DISTINCT trim(source_keyword) AS source_keyword FROM all_features WHERE source_keyword IS NOT NULL AND length(trim(source_keyword)) > 0 ORDER BY 1",
        "SELECT DISTINCT trim(source_keyword) AS source_keyword FROM post_comment_aggregate WHERE source_keyword IS NOT NULL AND length(trim(source_keyword)) > 0 ORDER BY 1",
        "SELECT DISTINCT trim(source_keyword) AS source_keyword FROM content_gap WHERE source_keyword IS NOT NULL AND length(trim(source_keyword)) > 0 ORDER BY 1",
    ]
    for sql in candidates:
        rows = conn.execute(sql).fetchall()
        keywords = [row[0] for row in rows if row and row[0]]
        if keywords:
            return keywords
    return []


def get_topic_data(keyword: str) -> Dict[str, pd.DataFrame]:
    conn = get_duckdb_conn()
    safe_keyword = keyword.replace("'", "''")
    all_features = conn.execute(
        f"""
        SELECT *
        FROM all_features
        WHERE lower(source_keyword) = lower('{safe_keyword}')
        """
    ).df()
    comment_agg = conn.execute(
        f"""
        SELECT *
        FROM post_comment_aggregate
        WHERE lower(source_keyword) = lower('{safe_keyword}')
        """
    ).df()
    content_gap = conn.execute(
        f"""
        SELECT *
        FROM content_gap
        WHERE lower(source_keyword) = lower('{safe_keyword}')
        """
    ).df()
    return {
        "all_features": all_features,
        "post_comment_aggregate": comment_agg,
        "content_gap": content_gap,
    }


def get_full_data() -> Dict[str, pd.DataFrame]:
    conn = get_duckdb_conn()
    all_features = conn.execute("SELECT * FROM all_features").df()
    comment_agg = conn.execute("SELECT * FROM post_comment_aggregate").df()
    content_gap = conn.execute("SELECT * FROM content_gap").df()
    return {
        "all_features": all_features,
        "post_comment_aggregate": comment_agg,
        "content_gap": content_gap,
    }
