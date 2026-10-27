# Roadmap

Prioritised by risk to users first, then by value. Effort: S ≈ days, M ≈ 1–2 weeks, L ≈ a month+.

## P0 — before any shared or networked deployment

| Item | Why | Effort |
|---|---|---|
| **Authentication & per-user workspaces** (OIDC via a reverse proxy header, or built-in sessions) with ownership checks on every workspace route | Today anyone who can reach the server can use it; workspace IDs are bearer tokens. | M |
| **Build and smoke-test the Docker image in CI**; publish a pinned image | The Dockerfile was written but not built in the verification environment. | S |
| **Test the live LLM path** against the real Anthropic API (routing, narration, timeouts, refusal handling) | It is covered only with a mocked API. | S |
| **CI pipeline**: backend tests on Python 3.10–3.13 × pandas 2.2/3.x, Vitest, Playwright, `pip-audit`, `npm audit`, Docker build | Prevent regressions; the verification here was manual. | S |

## P1 — reliability and scale

| Item | Why | Effort |
|---|---|---|
| Out-of-process job queue (e.g. RQ/Celery or `ProcessPoolExecutor`) and support for multiple workers | Training holds the GIL partly and jobs are lost on restart; only one uvicorn worker is supported. | M |
| Chunked/streaming ingestion and Polars/DuckDB for profiling > 2M rows | Whole files are held in memory; the 1M-row insights call takes ~10 s. | L |
| Persist job state and resume/mark interrupted jobs after restart | A restart during training leaves the job unknown. | S |
| Quotas per workspace (disk, number of models/datasets) | Disk use is bounded only by the TTL. | S |
| Probe-based time estimates for every model family (currently random forest / tree only) | Boosting models on very wide data can still overrun the budget. | S |

## P2 — analytics depth

| Item | Why | Effort |
|---|---|---|
| Model monitoring: data drift (PSI/KS) between training data and batch-scored files | Detect when predictions stop being trustworthy. | M |
| Probability calibration (isotonic/Platt) as a selectable step, with reliability plots | Calibration is reported but not corrected. | S |
| SHAP-style local explanations for tree models | Permutation importance is global; single-prediction explanations are approximate. | M |
| Forecasting with exogenous regressors and holiday effects; hierarchical/grouped forecasts | Common business needs. | M |
| Formal A/B test analysis module (power, sequential testing guidance) | Diagnostics currently treat observational comparisons only. | M |
| Join suggestions across more than two files; fuzzy key matching | Multi-file workflows beyond pairs. | M |

## P3 — product polish

| Item | Why | Effort |
|---|---|---|
| Scheduled report regeneration and email/Slack delivery | Recurring reporting. | M |
| Shareable read-only report links with expiry | Sharing without giving workspace access. | S |
| Saved dashboards built from the chart builder | Charts are recomputed per session today. | S |
| Internationalisation (number/date formats, translations) | Formats currently follow the browser locale only partly. | M |
| Toasts can briefly overlap content on narrow screens; move them to a dedicated lane | Minor visual issue seen in E2E screenshots. | S |
| Automated accessibility checks (axe) in Playwright | Accessibility was checked manually and by role-based selectors only. | S |
