"""
Safe, transparent file ingestion.

Supported formats: CSV / TSV / TXT (delimited), Excel (.xlsx, .xls), JSON and
JSON Lines. Every decision taken while reading the file (encoding, delimiter,
skipped malformed rows, renamed columns, lossless type conversions) is recorded
in an *ingestion report* that is shown to the user.

Design rules
------------
* Never silently drop or alter data. Malformed rows are counted and sampled;
  duplicate or blank column names are renamed and reported.
* Only *lossless* conversions happen at ingestion (e.g. a text column whose
  every value is a valid ISO date becomes a datetime). Lossy conversions are
  offered later as cleaning suggestions that require approval.
* Resource limits (bytes, rows, columns, cells) are enforced up front.
"""
from __future__ import annotations

import csv
import io
import json
import warnings
import zipfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.config import settings
from core.errors import PayloadTooLarge, UnsupportedMedia, ValidationFailed
from core.security import sanitize_filename, sha256_bytes

from .schema import is_text_dtype

SUPPORTED_EXTENSIONS = {"csv", "tsv", "txt", "xlsx", "xls", "json", "jsonl", "ndjson"}
DELIMITED = {"csv", "tsv", "txt"}
EXCEL = {"xlsx", "xls"}
JSONISH = {"json", "jsonl", "ndjson"}
CANDIDATE_DELIMITERS = [",", ";", "\t", "|"]
MAX_MALFORMED_SAMPLES = 5


async def read_upload(upload, max_bytes: Optional[int] = None) -> bytes:
    """Read an UploadFile in chunks, aborting as soon as the size limit is exceeded."""
    limit = max_bytes or settings.max_upload_bytes
    chunks: List[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise PayloadTooLarge(
                f"File exceeds the {limit // (1024 * 1024)} MB upload limit. "
                "Increase MAX_UPLOAD_MB in backend/.env if your machine has enough memory."
            )
        chunks.append(chunk)
    return b"".join(chunks)


def file_extension(filename: str) -> str:
    name = sanitize_filename(filename)
    if "." not in name:
        return ""
    return name.rsplit(".", 1)[-1].lower()


def check_format(content: bytes, filename: str) -> str:
    ext = file_extension(filename)
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedMedia(
            f"Unsupported file type '.{ext or '?'}'. Supported: CSV, TSV, TXT, XLSX, XLS, JSON, JSONL."
        )
    if len(content) == 0 or not content.strip():
        raise ValidationFailed("The file is empty.", code="empty_file")
    if ext == "xlsx" and not content.startswith(b"PK\x03\x04"):
        raise UnsupportedMedia("This .xlsx file is not a valid Excel workbook (wrong file signature).")
    if ext == "xls" and not content.startswith(b"\xd0\xcf\x11\xe0"):
        if content.startswith(b"PK\x03\x04"):
            raise UnsupportedMedia("This file is an .xlsx workbook saved with an .xls extension. Rename it to .xlsx.")
        raise UnsupportedMedia("This .xls file is not a valid legacy Excel workbook.")
    if ext in DELIMITED and (content.startswith(b"PK\x03\x04") or content.startswith(b"\xd0\xcf\x11\xe0")):
        raise UnsupportedMedia("This looks like an Excel workbook, not a text file. Use the .xlsx/.xls extension.")
    if ext in DELIMITED | JSONISH and b"\x00" in content[:4096] and not content.startswith((b"\xff\xfe", b"\xfe\xff")):
        raise UnsupportedMedia("The file appears to be binary, not text.")
    return ext


# ── Text decoding ────────────────────────────────────────────────────────────

def decode_text(content: bytes) -> Tuple[str, str]:
    """Decode bytes, returning (text, encoding). Tries strict decoders first."""
    if content.startswith((b"\xff\xfe", b"\xfe\xff")):
        return content.decode("utf-16"), "utf-16"
    for enc in ("utf-8-sig", "cp1252"):
        try:
            text = content.decode(enc)
            return text, ("utf-8" if enc == "utf-8-sig" else enc)
        except UnicodeDecodeError:
            continue
    return content.decode("latin-1"), "latin-1"


def detect_delimiter(text: str, ext: str) -> str:
    if ext == "tsv":
        return "\t"
    sample = "\n".join(text[:65536].splitlines()[:50])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="".join(CANDIDATE_DELIMITERS))
        if dialect.delimiter in CANDIDATE_DELIMITERS:
            # Validate the sniffer's choice with a consistency check below.
            sniffed = dialect.delimiter
        else:
            sniffed = None
    except csv.Error:
        sniffed = None

    lines = [ln for ln in sample.splitlines() if ln.strip()][:50]
    best, best_score = ",", -1.0
    for d in CANDIDATE_DELIMITERS:
        try:
            counts = [len(r) for r in csv.reader(lines, delimiter=d)]
        except csv.Error:
            continue
        if not counts or max(counts) <= 1:
            continue
        mode = max(set(counts), key=counts.count)
        consistency = counts.count(mode) / len(counts)
        score = consistency * 10 + min(mode, 50) / 50
        if score > best_score:
            best, best_score = d, score
    if sniffed and sniffed != best:
        # Prefer the sniffer only when it is at least as consistent.
        counts = [len(r) for r in csv.reader(lines, delimiter=sniffed)]
        if counts and max(counts) > 1:
            mode = max(set(counts), key=counts.count)
            if counts.count(mode) / len(counts) * 10 + min(mode, 50) / 50 >= best_score:
                return sniffed
    return best


# ── Column-name repair ───────────────────────────────────────────────────────

def repair_column_names(raw: List[Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Make names unique, non-empty, trimmed strings. Returns (names, changes)."""
    changes: List[Dict[str, Any]] = []
    names: List[str] = []
    seen: Dict[str, int] = {}
    for i, r in enumerate(raw):
        orig = "" if r is None or (isinstance(r, float) and np.isnan(r)) else str(r)
        name = " ".join(orig.split())  # trim + collapse internal whitespace
        if name.lower().startswith("unnamed:") and name[8:].strip().isdigit():
            name = ""
        if not name:
            name = f"column_{i + 1}"
            changes.append({"position": i + 1, "original": orig, "new": name, "reason": "blank header"})
        elif name != orig:
            changes.append({"position": i + 1, "original": orig, "new": name, "reason": "whitespace trimmed"})
        key = name.lower()
        if key in seen:
            seen[key] += 1
            new = f"{name}_{seen[key]}"
            while new.lower() in seen:
                seen[key] += 1
                new = f"{name}_{seen[key]}"
            changes.append({"position": i + 1, "original": orig, "new": new, "reason": "duplicate column name"})
            name = new
            seen[new.lower()] = 1
        else:
            seen[key] = 1
        names.append(name)
    return names, changes


# ── Readers ──────────────────────────────────────────────────────────────────

def _read_delimited(content: bytes, ext: str, report: Dict) -> pd.DataFrame:
    text, encoding = decode_text(content)
    report["encoding"] = encoding
    if encoding not in ("utf-8", "utf-16"):
        report["warnings"].append(
            f"File is not valid UTF-8; decoded as {encoding}. Check accented characters in the preview."
        )
    delim = detect_delimiter(text, ext)
    report["delimiter"] = {",": "comma", ";": "semicolon", "\t": "tab", "|": "pipe"}[delim]

    first = next(csv.reader(io.StringIO(text), delimiter=delim), None)
    if not first or all(not str(x).strip() for x in first):
        raise ValidationFailed("The file has no header row.", code="empty_file")
    names, changes = repair_column_names(first)
    report["column_changes"] = changes

    max_rows = settings.max_rows
    # index_col=False: never treat extra leading fields as an index (that silently shifts data).
    common = dict(sep=delim, header=0, names=names, nrows=max_rows + 1, skipinitialspace=False, index_col=False)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pd.read_csv(io.StringIO(text), engine="c", on_bad_lines="error", low_memory=False, **common)
    except pd.errors.ParserError:
        # Quote-aware pre-scan with the csv module: rows with more fields than the header are
        # removed and reported (pandas' own handling differs between versions and can truncate silently).
        width = len(names)
        bad: List[Dict[str, Any]] = []
        n_bad = 0
        clean = io.StringIO()
        writer = csv.writer(clean, delimiter=delim, lineterminator="\n")
        reader = csv.reader(io.StringIO(text), delimiter=delim)
        for i, row in enumerate(reader):
            if i > 0 and len(row) > width:
                n_bad += 1
                if len(bad) < MAX_MALFORMED_SAMPLES:
                    bad.append({"line": reader.line_num, "fields": len(row), "values": [str(f)[:60] for f in row[:12]]})
                continue
            writer.writerow(row)
        clean.seek(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pd.read_csv(clean, engine="c", low_memory=False, **common)
        report["malformed_rows"] = {"count": n_bad, "samples": bad, "action": "skipped", "expected_fields": width}
        report["warnings"].append(
            f"{n_bad} malformed row(s) had more fields than the header ({width}) and were skipped. "
            "Line numbers and values are shown in the ingestion report."
        )
    return df


def list_excel_sheets(content: bytes, ext: str) -> List[str]:
    _guard_zip(content, ext)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xf = pd.ExcelFile(io.BytesIO(content), engine="openpyxl" if ext == "xlsx" else "xlrd")
        return [str(s) for s in xf.sheet_names]


def _guard_zip(content: bytes, ext: str) -> None:
    """Protect against decompression bombs in .xlsx archives."""
    if ext != "xlsx":
        return
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            total = sum(i.file_size for i in zf.infolist())
    except zipfile.BadZipFile:
        raise UnsupportedMedia("The .xlsx file is corrupted (invalid zip container).")
    limit = min(max(len(content) * 200, 50 * 1024 * 1024), 2 * 1024 ** 3)
    if total > limit:
        raise PayloadTooLarge("The workbook expands to an unsafe size when decompressed and was rejected.")


def _read_excel(content: bytes, ext: str, sheet: Optional[str], report: Dict) -> pd.DataFrame:
    sheets = list_excel_sheets(content, ext)
    if not sheets:
        raise ValidationFailed("The workbook contains no sheets.", code="empty_file")
    report["sheets"] = sheets
    if sheet is None:
        sheet = sheets[0]
        if len(sheets) > 1:
            report["warnings"].append(
                f"Workbook has {len(sheets)} sheets; loaded '{sheet}'. Choose another sheet on upload if needed."
            )
    elif sheet not in sheets:
        raise ValidationFailed(f"Sheet '{sheet}' not found. Available: {', '.join(sheets)}")
    report["sheet"] = sheet
    engine = "openpyxl" if ext == "xlsx" else "xlrd"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        head = pd.read_excel(io.BytesIO(content), sheet_name=sheet, header=None, nrows=1, engine=engine)
        if head.empty:
            raise ValidationFailed(f"Sheet '{sheet}' is empty.", code="empty_file")
        names, changes = repair_column_names(head.iloc[0].tolist())
        report["column_changes"] = changes
        df = pd.read_excel(io.BytesIO(content), sheet_name=sheet, header=0, engine=engine,
                           nrows=settings.max_rows + 1)
    # Excel may return fewer header cells than data width; align safely.
    if len(df.columns) == len(names):
        df.columns = names
    else:
        df.columns, extra = repair_column_names(list(df.columns))
        report["column_changes"] = extra
    return df


def _read_json(content: bytes, ext: str, report: Dict) -> pd.DataFrame:
    text, encoding = decode_text(content)
    report["encoding"] = encoding
    records: Any
    if ext in ("jsonl", "ndjson"):
        records, bad = [], 0
        for ln in text.splitlines():
            if not ln.strip():
                continue
            try:
                records.append(json.loads(ln))
            except json.JSONDecodeError:
                bad += 1
        if bad:
            report["malformed_rows"] = {"count": bad, "samples": [], "action": "skipped"}
            report["warnings"].append(f"{bad} line(s) were not valid JSON and were skipped.")
    else:
        try:
            records = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValidationFailed(f"Invalid JSON: {e.msg} (line {e.lineno}, column {e.colno}).")
        if isinstance(records, dict):
            list_keys = [k for k, v in records.items() if isinstance(v, list)]
            if len(list_keys) == 1:
                report["warnings"].append(f"Loaded the records under the '{list_keys[0]}' key.")
                records = records[list_keys[0]]
            elif all(not isinstance(v, (list, dict)) for v in records.values()):
                records = [records]
            else:
                # dict of columns {col: {idx: val}} or {col: [..]}
                try:
                    return pd.DataFrame(records)
                except Exception:
                    raise ValidationFailed("Unsupported JSON layout. Provide an array of records.")
    if not isinstance(records, list):
        raise ValidationFailed("Unsupported JSON layout. Provide an array of records.")
    if len(records) == 0:
        raise ValidationFailed("The JSON file contains no records.", code="empty_file")
    if len(records) > settings.max_rows:
        raise PayloadTooLarge(f"The file has more than {settings.max_rows:,} rows (MAX_ROWS).")
    if not all(isinstance(r, dict) for r in records):
        if all(not isinstance(r, (dict, list)) for r in records):
            return pd.DataFrame({"value": records})
        raise ValidationFailed("Every JSON record must be an object with the same kind of fields.")
    df = pd.json_normalize(records, max_level=1)
    for c in df.columns:
        if df[c].map(lambda v: isinstance(v, (list, dict))).any():
            df[c] = df[c].map(lambda v: json.dumps(v) if isinstance(v, (list, dict)) else v)
            report["warnings"].append(f"Nested values in '{c}' were stored as JSON text.")
    names, changes = repair_column_names(list(df.columns))
    df.columns = names
    report["column_changes"] = changes
    return df


# ── Post-processing ──────────────────────────────────────────────────────────

def _lossless_datetime(df: pd.DataFrame, report: Dict) -> pd.DataFrame:
    """Convert text columns to datetime only when *every* non-null value parses
    with one consistent format. Anything less becomes a cleaning suggestion."""
    for col in df.columns:
        s = df[col]
        if not is_text_dtype(s):
            continue
        nn = s.dropna()
        if len(nn) == 0:
            continue
        smp = nn.astype(str).str.strip().head(200)
        if not smp.str.match(r"^\d{4}-\d{1,2}-\d{1,2}([ T]\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?)?(Z|[+-]\d{2}:?\d{2})?$").all() \
                and not smp.str.match(r"^\d{1,2}[/.-]\d{1,2}[/.-]\d{4}$").all():
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                parsed = pd.to_datetime(nn.astype(str).str.strip(), errors="coerce")
            except Exception:
                continue
        if parsed.notna().all():
            if getattr(parsed.dt, "tz", None) is not None:
                parsed = parsed.dt.tz_convert("UTC").dt.tz_localize(None)
            out = pd.Series(pd.NaT, index=s.index, dtype=parsed.dtype)
            out.loc[nn.index] = parsed
            df[col] = out
            ambiguous = smp.str.match(r"^\d{1,2}[/.-]\d{1,2}[/.-]\d{4}$").all()
            report["type_conversions"].append({
                "column": col, "from": "text", "to": "datetime", "lossless": True,
                "note": ("Day/month order inferred from the data; verify in the preview."
                         if ambiguous else "All values were valid dates."),
            })
    return df


def _normalise_object_columns(df: pd.DataFrame, report: Dict) -> pd.DataFrame:
    """Mixed-type object columns (common in Excel/JSON) are stored as text so they
    can be persisted and analysed consistently. Numbers-only object columns become numeric."""
    for col in df.columns:
        s = df[col]
        if not pd.api.types.is_object_dtype(s):
            continue
        nn = s.dropna()
        if len(nn) == 0:
            continue
        kinds = set(type(v).__name__ for v in nn)
        if kinds <= {"int", "float", "int64", "float64", "int32", "float32"}:
            df[col] = pd.to_numeric(s, errors="coerce")
        elif kinds <= {"bool", "bool_"}:
            df[col] = s.astype("boolean")
        elif kinds <= {"datetime", "Timestamp"}:
            df[col] = pd.to_datetime(s, errors="coerce")
        elif kinds <= {"str"}:
            continue
        else:
            df[col] = s.map(lambda v: v if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
            report["warnings"].append(
                f"Column '{col}' mixed several value types ({', '.join(sorted(kinds))}); stored as text."
            )
    return df


def load_dataframe(content: bytes, filename: str, sheet: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:
    ext = check_format(content, filename)
    report: Dict[str, Any] = {
        "filename": sanitize_filename(filename),
        "format": ext,
        "size_bytes": len(content),
        "sha256": sha256_bytes(content),
        "encoding": None,
        "delimiter": None,
        "sheet": None,
        "sheets": None,
        "malformed_rows": {"count": 0, "samples": [], "action": "none"},
        "column_changes": [],
        "type_conversions": [],
        "warnings": [],
    }
    try:
        if ext in DELIMITED:
            df = _read_delimited(content, ext, report)
        elif ext in EXCEL:
            df = _read_excel(content, ext, sheet, report)
        else:
            df = _read_json(content, ext, report)
    except (ValidationFailed, PayloadTooLarge, UnsupportedMedia):
        raise
    except pd.errors.EmptyDataError:
        raise ValidationFailed("The file is empty.", code="empty_file")
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as e:
        raise ValidationFailed(f"The file could not be parsed as {ext.upper()}: {str(e)[:200]}")
    except Exception as e:  # corrupted workbooks etc.
        raise ValidationFailed(f"The file could not be read ({type(e).__name__}). Is it a valid {ext.upper()} file?")

    if len(df.columns) == 0:
        raise ValidationFailed("No columns were found in the file.", code="empty_file")
    if len(df) == 0:
        raise ValidationFailed("The file has a header but no data rows.", code="empty_file")
    if len(df) > settings.max_rows:
        raise PayloadTooLarge(
            f"The file has more than {settings.max_rows:,} rows, the configured MAX_ROWS limit. "
            "Split the file or raise the limit in backend/.env."
        )
    if len(df.columns) > settings.max_columns:
        raise PayloadTooLarge(
            f"The file has {len(df.columns):,} columns; the limit is {settings.max_columns:,} (MAX_COLUMNS)."
        )
    if df.shape[0] * df.shape[1] > settings.max_cells:
        raise PayloadTooLarge(f"The file has {df.shape[0] * df.shape[1]:,} cells; the limit is {settings.max_cells:,}.")

    # Fully empty rows are reported (not removed) – they are a cleaning decision.
    empty_rows = int(df.isna().all(axis=1).sum())
    if empty_rows:
        report["warnings"].append(f"{empty_rows} completely empty row(s) found. You can remove them in the Clean step.")
    report["empty_rows"] = empty_rows
    if len(df.columns) > 200:
        report["warnings"].append(f"Wide dataset ({len(df.columns)} columns): some charts show a subset of columns.")

    df = _normalise_object_columns(df, report)
    df = _lossless_datetime(df, report)
    df = df.reset_index(drop=True)
    report["rows"] = int(len(df))
    report["columns"] = int(len(df.columns))
    return df, report


def inspect_file(content: bytes, filename: str) -> Dict[str, Any]:
    """Lightweight pre-check used by the upload wizard (sheet selection etc.)."""
    ext = check_format(content, filename)
    info: Dict[str, Any] = {"filename": sanitize_filename(filename), "format": ext, "size_bytes": len(content)}
    if ext in EXCEL:
        info["sheets"] = list_excel_sheets(content, ext)
    elif ext in DELIMITED:
        text, enc = decode_text(content[: 256 * 1024])
        delim = detect_delimiter(text, ext)
        info["encoding"] = enc
        info["delimiter"] = {",": "comma", ";": "semicolon", "\t": "tab", "|": "pipe"}[delim]
        rows = list(csv.reader(io.StringIO("\n".join(text.splitlines()[:6])), delimiter=delim))
        info["header"] = rows[0] if rows else []
        info["sample_rows"] = rows[1:6]
    return info
