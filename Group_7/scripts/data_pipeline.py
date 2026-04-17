import argparse
import hashlib
import json
import time
from pathlib import Path

from rag_engine import init_vector_store


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data_sourse"
STATE_PATH = PROJECT_ROOT / ".pipeline_state.json"
CSV_FILES = [
    DATA_DIR / "all_features.csv",
    DATA_DIR / "post_comment_aggregate.csv",
    DATA_DIR / "content_gap.csv",
]


def _file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _snapshot() -> dict:
    snapshot = {}
    for file_path in CSV_FILES:
        if file_path.exists():
            snapshot[str(file_path)] = _file_hash(file_path)
        else:
            snapshot[str(file_path)] = ""
    return snapshot


def _load_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _validate_inputs() -> None:
    missing = [str(p) for p in CSV_FILES if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required CSV files: {missing}")


def run_once(max_docs: int, force: bool) -> bool:
    _validate_inputs()
    current = _snapshot()
    previous = _load_state().get("csv_hash", {})
    changed = current != previous

    if not force and not changed:
        print("No CSV changes detected. Skipping vector rebuild.")
        return False

    ok, msg = init_vector_store(max_docs=max_docs, rebuild=True)
    print(msg)
    if not ok:
        raise RuntimeError("Vector rebuild failed.")

    _save_state({"csv_hash": current, "last_updated": int(time.time())})
    print("Pipeline state updated.")
    return True


def watch(interval: int, max_docs: int) -> None:
    print(f"Watching CSV changes every {interval}s...")
    while True:
        try:
            changed = run_once(max_docs=max_docs, force=False)
            if changed:
                print("Detected updates and rebuilt vector index.")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Pipeline error: {exc}")
        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Data pipeline for CSV sync and vector rebuild")
    parser.add_argument("--max-docs", type=int, default=50000, help="Max docs to embed for vector rebuild")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if CSV unchanged")
    parser.add_argument("--watch", action="store_true", help="Watch mode for near real-time updates")
    parser.add_argument("--interval", type=int, default=300, help="Watch interval in seconds")
    args = parser.parse_args()

    if args.watch:
        watch(interval=args.interval, max_docs=args.max_docs)
    else:
        run_once(max_docs=args.max_docs, force=args.force)


if __name__ == "__main__":
    main()
