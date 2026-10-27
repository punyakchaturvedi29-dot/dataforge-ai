# Security and privacy

DataForge is designed as a **single-user or small-team tool run on a trusted machine or network**.
It has no user accounts. This document states what it protects against, what it does not, and how
data is stored and deleted.

## Threat model

| In scope (mitigated) | Out of scope (needs deployment controls) |
|---|---|
| Malicious or malformed files (oversized, disguised, broken encodings, pathological CSV) | Authentication / authorisation of users |
| Path traversal through workspace, dataset, model or prediction IDs | Multi-tenant isolation between untrusted users |
| CSV/Excel formula injection in exported files | TLS termination (use a reverse proxy) |
| Execution of untrusted serialized models | Rate limiting / DoS from the network |
| Arbitrary code execution through the copilot or LLM output | Encryption at rest (use an encrypted volume) |
| Leaking stack traces, secrets or file paths in API responses | |
| Catastrophic regular expressions in quality rules | |
| Unbounded memory/CPU use by uploads, JSON bodies and model training | |

**Do not expose the server directly to the internet.** Anyone who can reach it can upload data and
read any workspace whose ID they know (IDs are 128-bit random values; with `WORKSPACE_LISTING=false`
they are not enumerable through the API). For shared use, put it behind a reverse proxy that performs
authentication (e.g. OAuth2 Proxy, an identity-aware proxy, or HTTP basic auth over TLS).

## Controls

- **Uploads:** size limit enforced while streaming and via `Content-Length`; row, column and cell
  limits; extension and content-sniffing checks (e.g. a ZIP renamed to `.csv` is refused); file names
  sanitised; original bytes hashed (SHA-256) for provenance.
- **Identifiers and paths:** every ID is normalised to lowercase hex and every file path is resolved
  with `safe_child`, which refuses anything outside the workspace directory.
- **Models:** only models trained by this server are loaded. Artifacts are signed with HMAC-SHA256
  (`DATAFORGE_SECRET_KEY` or a generated key stored with mode 600) and verified before every load;
  a modified file is refused with `integrity_check_failed`. There is no model upload endpoint.
- **No code execution:** the copilot maps questions to a fixed set of analysis functions. LLM output
  is treated as text; tool calls proposed by the LLM must name a tool from the fixed list, and the tool
  resolves column names against the dataset (unknown names lead to a clarifying question).
- **Exports:** cells starting with `=`, `+`, `-`, `@`, tab or carriage return are prefixed with `'` in
  CSV/Excel downloads (`CSV_FORMULA_ESCAPE`); report PDFs escape markup in user-supplied names.
- **Errors:** a uniform JSON error shape with a request ID; unexpected exceptions are logged
  server-side and return a generic message.
- **HTTP:** CORS limited to `CORS_ORIGINS` and the methods the app uses; `X-Content-Type-Options:
  nosniff`, `Referrer-Policy: no-referrer`, `Cache-Control: no-store` on API responses; JSON body
  size limit (`MAX_JSON_BODY_KB`).
- **Secrets:** read from the environment or `backend/.env` (git-ignored). `/api/config` and
  `/api/health` expose only non-secret settings (tested). The API key is never sent to the browser
  or written to logs.
- **Resources:** background jobs run in a bounded pool (`MAX_CONCURRENT_JOBS`, `MAX_QUEUED_JOBS`);
  training is capped by `TRAINING_MAX_ROWS` and `TRAINING_TIME_BUDGET_S`; batch prediction by
  `BATCH_PREDICT_MAX_ROWS`.
- **Dependencies:** `pip-audit` and `npm audit` reported no known vulnerabilities at release
  (see the verification report). Re-run them regularly.

## Data handling and retention

- Everything is stored under `DATAFORGE_DATA_DIR` (default `backend/data`):
  `workspaces/<id>/` holds metadata JSON, datasets as parquet (original + cached working copy),
  models, predictions, chat history and saved analyses.
- Workspaces not updated for `WORKSPACE_TTL_DAYS` (default 14) are deleted automatically (checked at
  most hourly). Deleting a workspace in the UI or via `DELETE /api/workspaces/{id}` removes its
  directory immediately.
- Temporary files are written in the workspace directory and replaced atomically.
- **LLM privacy:** with `ANTHROPIC_API_KEY` set, the copilot sends the question, column names/types
  and the *computed result* of the selected tool to the Anthropic API — mostly aggregates, but questions
  such as "top 10 orders by revenue" include those rows (capped at about 12 KB of JSON).
  Set `LLM_ENABLED=false` or leave the key empty to keep everything local.
- The browser stores only preferences, the recent-workspace list and notifications in `localStorage`.
