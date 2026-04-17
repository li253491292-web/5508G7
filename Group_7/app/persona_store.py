import json
from datetime import datetime, timezone
from pathlib import Path


STORE_PATH = Path(__file__).resolve().parents[1] / "data_sourse" / "persona_cache.json"


def _read_store() -> dict:
    if not STORE_PATH.exists():
        return {}
    try:
        return json.loads(STORE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_store(payload: dict) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STORE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def get_persona(keyword: str) -> str:
    store = _read_store()
    entry = store.get((keyword or "").strip().lower(), {})
    return entry.get("persona_text", "")


def upsert_persona(keyword: str, persona_text: str) -> None:
    key = (keyword or "").strip().lower()
    if not key or not persona_text.strip():
        return
    store = _read_store()
    store[key] = {
        "keyword": keyword,
        "persona_text": persona_text,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_store(store)
