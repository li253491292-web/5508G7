from pathlib import Path
from typing import Tuple
import sqlite3

import faiss
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

from app.data_engine import PROJECT_ROOT, get_duckdb_conn


VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
FAISS_INDEX_PATH = VECTOR_DB_DIR / "all_features_full_text.faiss"
META_PATH = VECTOR_DB_DIR / "all_features_meta.parquet"
DATA_SOURCE_DIR = PROJECT_ROOT / "data_sourse"
CHROMA_SQLITE_CANDIDATES = [
    DATA_SOURCE_DIR / "chroma_db_master_china_medical.sqlite3",
    DATA_SOURCE_DIR / "chroma_db_features.sqlite3",
]
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@st.cache_resource
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)


def _candidate_vector_dirs() -> list[Path]:
    return [
        PROJECT_ROOT / "data_sourse",
        PROJECT_ROOT / "data_sourse" / "vector_db",
        PROJECT_ROOT / "vector_db",
    ]


def find_existing_vector_store() -> Tuple[Path, Path] | None:
    for directory in _candidate_vector_dirs():
        if not directory.exists():
            continue
        # Preferred canonical filenames.
        canonical_faiss = directory / "all_features_full_text.faiss"
        canonical_meta = directory / "all_features_meta.parquet"
        if canonical_faiss.exists() and canonical_meta.exists():
            return canonical_faiss, canonical_meta

        # Fallback: any .faiss + any .parquet in the same folder.
        faiss_files = sorted(directory.glob("*.faiss"))
        parquet_files = sorted(directory.glob("*.parquet"))
        if faiss_files and parquet_files:
            return faiss_files[0], parquet_files[0]
    return None


def vector_store_exists() -> bool:
    return find_existing_vector_store() is not None


def find_chroma_sqlite_files() -> list[Path]:
    existing = [p for p in CHROMA_SQLITE_CANDIDATES if p.exists()]
    if existing:
        return existing
    if DATA_SOURCE_DIR.exists():
        return sorted(DATA_SOURCE_DIR.glob("*.sqlite3"))
    return []


def init_vector_store(max_docs: int = 50000, rebuild: bool = False) -> Tuple[bool, str]:
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    if vector_store_exists() and not rebuild:
        existing = find_existing_vector_store()
        if existing:
            return True, f"Using existing vector DB: {existing[0].parent}"
        return True, "Vector DB already exists."

    conn = get_duckdb_conn()
    df = conn.execute(
        f"""
        SELECT post_id, source_keyword, subreddit, coalesce(full_text, '') AS full_text
        FROM all_features
        WHERE full_text IS NOT NULL AND length(trim(full_text)) > 0
        LIMIT {int(max_docs)}
        """
    ).df()
    if df.empty:
        return False, "No usable full_text records found."

    embedder = get_embedder()
    vectors = embedder.encode(df["full_text"].tolist(), show_progress_bar=True)
    vectors = np.array(vectors).astype("float32")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    df.to_parquet(META_PATH, index=False)
    return True, f"Built vector DB with {len(df)} documents."


@st.cache_resource
def load_vector_store(index_path: str, meta_path: str) -> Tuple[faiss.Index, pd.DataFrame]:
    index = faiss.read_index(index_path)
    meta = pd.read_parquet(meta_path)
    return index, meta


def retrieve_corpus_faiss(query_text: str, top_k: int = 10) -> pd.DataFrame:
    resolved = find_existing_vector_store()
    if not resolved:
        return pd.DataFrame()
    index_path, meta_path = resolved
    index, meta = load_vector_store(str(index_path), str(meta_path))
    embedder = get_embedder()
    q = embedder.encode([query_text]).astype("float32")
    faiss.normalize_L2(q)
    scores, ids = index.search(q, top_k)
    rows = []
    for idx, score in zip(ids[0], scores[0]):
        if idx < 0 or idx >= len(meta):
            continue
        item = meta.iloc[int(idx)].to_dict()
        item["similarity"] = float(score)
        rows.append(item)
    return pd.DataFrame(rows)


def retrieve_corpus_chroma(query_text: str, top_k: int = 10, sqlite_path: str = "") -> pd.DataFrame:
    paths = [Path(sqlite_path)] if sqlite_path else find_chroma_sqlite_files()
    rows = []
    per_db_limit = max(1, int(top_k / max(1, len(paths))))

    for db_path in paths:
        if not db_path.exists():
            continue
        try:
            con = sqlite3.connect(str(db_path))
            cur = con.cursor()
            result = cur.execute(
                """
                SELECT
                    rowid,
                    string_value,
                    bm25(embedding_fulltext_search) AS rank_score
                FROM embedding_fulltext_search
                WHERE embedding_fulltext_search MATCH ?
                ORDER BY rank_score
                LIMIT ?
                """,
                (query_text, per_db_limit),
            ).fetchall()
            for rowid, text, rank_score in result:
                rows.append(
                    {
                        "post_id": f"{db_path.stem}:{rowid}",
                        "source_keyword": "chroma",
                        "subreddit": db_path.stem,
                        "full_text": text,
                        # sqlite bm25 is lower=better; flip sign for consistency.
                        "similarity": float(-rank_score),
                    }
                )
        except Exception:
            continue
        finally:
            try:
                con.close()
            except Exception:
                pass

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("similarity", ascending=False).head(top_k).reset_index(drop=True)
    return df


def retrieve_corpus(
    query_text: str,
    top_k: int = 10,
    backend: str = "auto",
    chroma_sqlite_path: str = "",
) -> pd.DataFrame:
    chosen = backend.lower().strip()
    if chosen == "faiss":
        return retrieve_corpus_faiss(query_text=query_text, top_k=top_k)
    if chosen == "chroma":
        return retrieve_corpus_chroma(query_text=query_text, top_k=top_k, sqlite_path=chroma_sqlite_path)

    # auto: prefer Chroma sqlite if present, otherwise FAISS.
    if find_chroma_sqlite_files():
        chroma_df = retrieve_corpus_chroma(query_text=query_text, top_k=top_k, sqlite_path=chroma_sqlite_path)
        if not chroma_df.empty:
            return chroma_df
    return retrieve_corpus_faiss(query_text=query_text, top_k=top_k)
