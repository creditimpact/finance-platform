import random
import time
from typing import Any, Dict, Tuple

_MIN_TTL_SEC = 14 * 24 * 60 * 60
_MAX_TTL_SEC = 30 * 24 * 60 * 60

CacheKey = Tuple[str, str, str, str]
CacheValue = Tuple[Dict[str, Any], float, int, int]

_CACHE: Dict[CacheKey, CacheValue] = {}

def _now() -> float:
    return time.time()

def get_cached_analysis(
    doc_fingerprint: str,
    bureau: str,
    prompt_hash: str,
    model_version: str,
    *,
    prompt_version: int,
    schema_version: int,
) -> Dict[str, Any] | None:
    key = (doc_fingerprint, bureau, prompt_hash, model_version)
    item = _CACHE.get(key)
    if not item:
        return None
    data, expires, p_version, s_version = item
    if _now() > expires or p_version != prompt_version or s_version != schema_version:
        _CACHE.pop(key, None)
        return None
    return data

def store_cached_analysis(
    doc_fingerprint: str,
    bureau: str,
    prompt_hash: str,
    model_version: str,
    result: Dict[str, Any],
    *,
    prompt_version: int,
    schema_version: int,
) -> None:
    ttl = random.randint(_MIN_TTL_SEC, _MAX_TTL_SEC)
    expires = _now() + ttl
    key = (doc_fingerprint, bureau, prompt_hash, model_version)
    _CACHE[key] = (result, expires, prompt_version, schema_version)

def reset_cache() -> None:
    _CACHE.clear()
