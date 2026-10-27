"""Upload, validation and type inference."""
import io
import json

import numpy as np
import pandas as pd
import pytest

from agents.ingest import load_dataframe
from agents.schema import infer_schema, semantic_type
from core.config import settings
from core.errors import PayloadTooLarge, UnsupportedMedia, ValidationFailed

from .conftest import csv_bytes, upload


def test_sample_upload_creates_workspace(client):
    r = upload(client, (pytest.importorskip("pathlib").Path(__file__).parents[2] / "sample_data.csv").read_bytes(),
               "sample_data.csv")
    assert r.status_code == 201
    body = r.json()
    ds = body["active_dataset"]
    assert ds["rows"] == 100 and ds["columns"] == 15
    assert ds["ingestion"]["encoding"] == "utf-8" and ds["ingestion"]["delimiter"] == "comma"
    conv = {c["column"]: c for c in ds["ingestion"]["type_conversions"]}
    assert conv["order_date"]["to"] == "datetime" and conv["order_date"]["lossless"]
    stages = {s["id"]: s["completed"] for s in body["stages"]}
    assert stages["upload"] and stages["validate"] and not stages["model"]


@pytest.mark.parametrize("sep,label", [(";", "semicolon"), ("\t", "tab"), ("|", "pipe"), (",", "comma")])
def test_delimiter_detection(sep, label):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [1.5, 2.5, 3.5]})
    out, rep = load_dataframe(csv_bytes(df, sep=sep), "f.csv")
    assert rep["delimiter"] == label
    assert list(out.columns) == ["a", "b", "c"] and len(out) == 3


def test_encodings():
    text = "name,city\nJosé,Zürich\nRené,Málaga\n"
    for enc, expected in (("cp1252", "cp1252"), ("utf-8-sig", "utf-8"), ("utf-16", "utf-16")):
        df, rep = load_dataframe(text.encode(enc), "f.csv")
        assert rep["encoding"] == expected
        assert df["city"].tolist() == ["Zürich", "Málaga"]


def test_malformed_rows_are_counted_and_skipped():
    content = b"a,b,c\n1,2,3\n4,5,6,7,8\n9,10,11\n"
    df, rep = load_dataframe(content, "f.csv")
    assert len(df) == 2
    assert rep["malformed_rows"]["count"] == 1
    s = rep["malformed_rows"]["samples"][0]
    assert s["values"][:3] == ["4", "5", "6"] and s["line"] == 3 and s["fields"] == 5


def test_duplicate_and_blank_headers_are_repaired():
    content = b"id,Value,value,,name\n1,2,3,4,x\n"
    df, rep = load_dataframe(content, "f.csv")
    assert list(df.columns) == ["id", "Value", "value_2", "column_4", "name"]
    reasons = {c["reason"] for c in rep["column_changes"]}
    assert {"duplicate column name", "blank header"} <= reasons


@pytest.mark.parametrize("content,code", [(b"", "empty_file"), (b"   \n ", "empty_file"), (b"a,b,c\n", "empty_file")])
def test_empty_files(content, code):
    with pytest.raises(ValidationFailed) as e:
        load_dataframe(content, "f.csv")
    assert e.value.code == code


def test_unsupported_and_disguised_files():
    with pytest.raises(UnsupportedMedia):
        load_dataframe(b"hello", "f.exe")
    with pytest.raises(UnsupportedMedia):
        load_dataframe(b"PK\x03\x04garbage", "f.csv")
    with pytest.raises(UnsupportedMedia):
        load_dataframe(b"not excel", "f.xlsx")
    with pytest.raises(UnsupportedMedia):
        load_dataframe(b"\x00\x01\x02binary\x00", "f.csv")


def test_excel_sheets_and_selection(client):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame({"a": [1, 2]}).to_excel(xw, sheet_name="First", index=False)
        pd.DataFrame({"b": [3, 4, 5], "c": ["x", "y", "z"]}).to_excel(xw, sheet_name="Second", index=False)
    content = buf.getvalue()
    r = client.post("/api/workspaces/inspect", files={"file": ("book.xlsx", content)})
    assert r.status_code == 200 and r.json()["sheets"] == ["First", "Second"]
    r = upload(client, content, "book.xlsx", sheet="Second")
    assert r.status_code == 201
    ds = r.json()["active_dataset"]
    assert ds["rows"] == 3 and ds["ingestion"]["sheet"] == "Second"
    r = upload(client, content, "book.xlsx", sheet="Nope")
    assert r.status_code == 422


def test_json_variants():
    df, _ = load_dataframe(json.dumps([{"a": 1, "b": {"c": 2}}, {"a": 3, "b": {"c": 4}}]).encode(), "f.json")
    assert list(df.columns) == ["a", "b.c"]
    df, rep = load_dataframe(json.dumps({"data": [{"x": 1}, {"x": 2}]}).encode(), "f.json")
    assert len(df) == 2 and any("data" in w for w in rep["warnings"])
    df, rep = load_dataframe(b'{"x": 1}\n{"x": 2}\nnot json\n', "f.jsonl")
    assert len(df) == 2 and rep["malformed_rows"]["count"] == 1
    with pytest.raises(ValidationFailed):
        load_dataframe(b"{bad json", "f.json")


def test_row_and_size_limits(client, monkeypatch):
    monkeypatch.setattr(settings, "max_rows", 10)
    with pytest.raises(PayloadTooLarge):
        load_dataframe(csv_bytes(pd.DataFrame({"a": range(11)})), "f.csv")
    monkeypatch.setattr(settings, "max_upload_mb", 1)
    big = b"a,b\n" + b"1,2\n" * 400_000
    r = upload(client, big, "big.csv")
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "payload_too_large"


def test_wide_dataset_is_accepted():
    df = pd.DataFrame(np.random.default_rng(0).normal(size=(5, 300)), columns=[f"c{i}" for i in range(300)])
    out, rep = load_dataframe(csv_bytes(df), "wide.csv")
    assert out.shape == (5, 300) and any("Wide dataset" in w for w in rep["warnings"])


def test_invalid_dates_are_not_converted_silently():
    content = b"d,v\n2024-01-01,1\n2024-13-45,2\n2024-02-01,3\n"
    df, rep = load_dataframe(content, "f.csv")
    assert not pd.api.types.is_datetime64_any_dtype(df["d"])
    assert rep["type_conversions"] == []


def test_logical_type_inference():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "order_id": np.arange(1, n + 1), "price": rng.normal(10, 2, n), "flag": rng.choice([0, 1], n),
        "yes_no": rng.choice(["yes", "no"], n), "city": rng.choice(["A", "B", "C"], n),
        "code": [f"X{i:04d}" for i in range(n)], "comment": ["this is a long free text comment number %d" % i for i in range(n)],
        "constant": ["k"] * n, "empty": [None] * n, "when": pd.date_range("2024-01-01", periods=n),
    })
    t = {c["name"]: c["logical_type"] for c in infer_schema(df)}
    assert t == {"order_id": "identifier", "price": "numeric", "flag": "boolean", "yes_no": "boolean",
                 "city": "categorical", "code": "identifier", "comment": "text", "constant": "constant",
                 "empty": "empty", "when": "datetime"}


@pytest.mark.parametrize("name,sem", [("discount_pct", "percentage"), ("sales_channel", "category"),
                                      ("delivery_days", "duration"), ("unit_price", "revenue"), ("customerAge", "age"),
                                      ("order_date", "date"), ("is_active", "boolean"), ("region", "location")])
def test_semantic_types_use_tokens_not_substrings(name, sem):
    assert semantic_type(name) == sem
