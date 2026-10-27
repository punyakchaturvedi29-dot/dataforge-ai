"""Security helpers: identifiers, filenames, CSV formula escaping, artifact signing."""
from __future__ import annotations

import hashlib
import hmac
import re
import unicodedata
import uuid
from pathlib import Path

import pandas as pd

from .errors import ValidationFailed, NotFound

_ID_RE = re.compile(r"^[a-f0-9]{8,64}$")
_LEGACY_UUID_RE = re.compile(r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$")


def new_id(n_hex: int = 32) -> str:
    """Random, unguessable identifier (128 bits by default)."""
    return uuid.uuid4().hex[:n_hex] if n_hex <= 32 else (uuid.uuid4().hex + uuid.uuid4().hex)[:n_hex]


def normalize_id(value: str, kind: str = "resource") -> str:
    """Validate an identifier before it is ever used to build a filesystem path.

    Accepts hex ids and (for v4 compatibility) dashed UUIDs, which are normalised
    to plain hex. Anything else is rejected, which rules out path traversal.
    """
    if not isinstance(value, str):
        raise NotFound(f"Unknown {kind}")
    v = value.strip().lower()
    if _LEGACY_UUID_RE.match(v):
        v = v.replace("-", "")
    if not _ID_RE.match(v):
        raise NotFound(f"Unknown {kind}")
    return v


def safe_child(base: Path, *parts: str) -> Path:
    """Join path parts and guarantee the result stays inside ``base``."""
    base_r = base.resolve()
    p = base_r.joinpath(*parts).resolve()
    if base_r != p and base_r not in p.parents:
        raise ValidationFailed("Invalid path")
    return p


_CTRL = re.compile(r"[\x00-\x1f\x7f]")
_BAD = re.compile(r'[\\/:*?"<>|]')


def sanitize_filename(name: str, default: str = "dataset") -> str:
    """Display-safe filename: no directories, control characters or reserved characters.

    Uploaded file names are only ever *displayed*; they are never used to build
    paths on disk.
    """
    if not name:
        return default
    name = unicodedata.normalize("NFKC", str(name))
    name = name.replace("\\", "/").split("/")[-1]
    name = _CTRL.sub("", name)
    name = _BAD.sub("_", name).strip().strip(".")
    if not name:
        return default
    if len(name) > 120:
        stem, dot, ext = name.rpartition(".")
        name = (stem[: 110] + "." + ext[:8]) if dot else name[:120]
    return name


def content_disposition(filename: str) -> str:
    """RFC 6266 header value with an ASCII fallback (prevents header injection)."""
    safe = sanitize_filename(filename, "download")
    ascii_name = safe.encode("ascii", "ignore").decode() or "download"
    ascii_name = ascii_name.replace('"', "")
    from urllib.parse import quote
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(safe)}"


_FORMULA_PREFIX = ("=", "+", "-", "@", "\t", "\r")


def escape_formula_value(v):
    if isinstance(v, str) and v.startswith(_FORMULA_PREFIX):
        return "'" + v
    return v


def escape_formulas(df: pd.DataFrame) -> pd.DataFrame:
    """Neutralise spreadsheet formula injection in text cells (OWASP guidance).

    Only text columns are touched; numeric columns (e.g. -5) are left intact.
    Column headers are escaped too.
    """
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            out[col] = s.map(escape_formula_value, na_action="ignore")
    out.columns = [escape_formula_value(str(c)) for c in out.columns]
    return out


def sign_bytes(data: bytes, secret: bytes) -> str:
    return hmac.new(secret, data, hashlib.sha256).hexdigest()


def verify_signature(data: bytes, signature: str, secret: bytes) -> bool:
    return hmac.compare_digest(sign_bytes(data, secret), (signature or "").strip())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- Regular-expression screening for user-defined data-quality rules ---------------------
_NESTED_QUANT = re.compile(r"\((?:[^()\\]|\\.)*[+*}](?:[^()\\]|\\.)*\)\s*[+*{]")


def check_user_regex(pattern: str) -> re.Pattern:
    """Compile a user-supplied regex with guards against catastrophic backtracking.

    Python's ``re`` has no timeout, so patterns are limited in length and nested
    quantifiers such as ``(a+)+`` (the classic ReDoS shape) are rejected.
    """
    if not isinstance(pattern, str) or not pattern:
        raise ValidationFailed("Pattern must be a non-empty string")
    if len(pattern) > 200:
        raise ValidationFailed("Pattern is too long (max 200 characters)")
    if _NESTED_QUANT.search(pattern):
        raise ValidationFailed("Nested quantifiers like (a+)+ are not allowed (they can hang the server)")
    if re.search(r"\\[1-9]", pattern):
        raise ValidationFailed("Back-references are not allowed in rule patterns")
    try:
        return re.compile(pattern)
    except re.error as e:
        raise ValidationFailed(f"Invalid regular expression: {e}")
