"""v4-compatible advanced analytics endpoints (``/api/advanced/*``)."""
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from agents.ml_advanced import _legacy, run_anomaly_detection, run_clustering
from core.errors import ValidationFailed
from services import analysis as A

from ._deps import get_ws

router = APIRouter()


class AdvancedPredictRequest(BaseModel):
    session_id: str
    task: str
    target_col: Optional[str] = None
    n_clusters: int = Field(3, ge=2, le=12)
    cluster_method: str = "kmeans"
    contamination: float = Field(0.05, ge=0.001, le=0.5)


@router.post("/predict")
def advanced_predict(req: AdvancedPredictRequest):
    df = get_ws(req.session_id).active.working
    if req.task in ("regression", "classify"):
        if not req.target_col:
            raise ValidationFailed("target_col required")
        return _legacy(df, req.target_col, "regression" if req.task == "regression" else "auto")
    if req.task == "cluster":
        return run_clustering(df, req.n_clusters, req.cluster_method)
    if req.task == "anomaly":
        return run_anomaly_detection(df, req.contamination)
    raise ValidationFailed(f"Unknown task: {req.task}")


@router.get("/profile/{session_id}")
def deep_profile(session_id: str):
    ws = get_ws(session_id)
    return A.profile(ws)


@router.get("/correlations/{session_id}")
def advanced_correlations(session_id: str):
    ws = get_ws(session_id)
    return A.corr(ws)
