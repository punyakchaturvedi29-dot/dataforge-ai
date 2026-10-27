"""v4-compatible quick prediction endpoint (``POST /api/predict/``).

Uses the AutoML engine in *quick* mode (4 candidates, no tuning) so results are
selected by cross-validation and reported on a held-out test set. The v4
row-order "forecast" was removed because it was not a valid forecast.
"""
from fastapi import APIRouter
from pydantic import BaseModel, Field

from agents.ml_advanced import _legacy, run_clustering
from core.errors import ValidationFailed

from ._deps import get_ws

router = APIRouter()


class PredictRequest(BaseModel):
    session_id: str
    task: str
    target_col: str = ""
    n_clusters: int = Field(3, ge=2, le=12)


@router.post("/")
def predict(req: PredictRequest):
    df = get_ws(req.session_id).active.working
    if req.task == "cluster":
        return run_clustering(df, req.n_clusters)
    if req.task not in ("regression", "classify"):
        raise ValidationFailed(f"Unknown task: {req.task}")
    if not req.target_col:
        raise ValidationFailed("target_col is required")
    return _legacy(df, req.target_col, "regression" if req.task == "regression" else "auto")
