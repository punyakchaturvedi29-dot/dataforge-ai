"""Forecasting and diagnostics (root-cause analysis)."""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from agents.forecasting import forecast_options, run_forecast
from agents.rca import run_diagnostics
from services import analysis as A

from ._deps import get_ws

router = APIRouter()


@router.get("/{workspace_id}/forecast/options")
def options(workspace_id: str):
    ws = get_ws(workspace_id)
    ds = ws.active
    return forecast_options(ds.working, A.schema(ws, ds))


class ForecastBody(BaseModel):
    date_column: str = Field(..., max_length=300)
    target: Optional[str] = Field(None, max_length=300, description="Numeric column; omit to forecast record counts")
    horizon: int = Field(6, ge=1, le=365)
    aggregation: Optional[str] = None
    frequency: Optional[str] = "auto"
    fill: str = "auto"
    methods: Optional[List[str]] = None
    backtest_folds: int = Field(3, ge=1, le=6)


@router.post("/{workspace_id}/forecast")
def forecast(workspace_id: str, body: ForecastBody):
    ws = get_ws(workspace_id)
    ds = ws.active
    r = run_forecast(ds.working, body.date_column, body.target, body.horizon, body.aggregation, body.frequency,
                     body.fill, body.methods, body.backtest_folds, A.schema(ws, ds))
    if r.get("available"):
        A.save_analysis(ws, "forecast", body.model_dump(), r, ds)
        ws.log("forecast", f"Forecast {r['aggregation']} of '{body.target or 'records'}' with {r['selected_label']}")
        ws.mark_stage("analyze", save=False)
        ws.save()
    return r


class DiagnosticsBody(BaseModel):
    outcome: str = Field(..., max_length=300)
    positive_class: Optional[str] = Field(None, max_length=300)
    dimensions: Optional[List[str]] = None
    date_column: Optional[str] = None
    period_a: Optional[List[str]] = Field(None, min_length=2, max_length=2)
    period_b: Optional[List[str]] = Field(None, min_length=2, max_length=2)


@router.post("/{workspace_id}/diagnostics")
def diagnostics(workspace_id: str, body: DiagnosticsBody):
    ws = get_ws(workspace_id)
    ds = ws.active
    r = run_diagnostics(ds.working, body.outcome, A.schema(ws, ds), body.positive_class, body.dimensions,
                        body.date_column, body.period_a, body.period_b)
    A.save_analysis(ws, "diagnostics", body.model_dump(), r, ds)
    ws.log("diagnostics", f"Diagnosed drivers of '{body.outcome}'")
    ws.mark_stage("analyze", save=False)
    ws.save()
    return r
