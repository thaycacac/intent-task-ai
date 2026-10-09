# Deploy guide — Railway (MVP)

Single cloud target for the Intent Task AI API. Prefer this over multi-cloud sprawl for the MLOps lab loop.

## Prerequisites

- Railway account + CLI (`npm i -g @railway/cli`) or dashboard
- Local artifacts at `artifacts/category/latest/` (run train scripts first)
- Repo pushed by **board** (agents do not `git push`)

## Steps

1. **Create project** on [railway.app](https://railway.app) → New Project → Deploy from GitHub (or empty + CLI).
2. **Root directory** = this repo root (Dockerfile present).
3. **Variables**

| Name | Value |
|------|-------|
| `ARTIFACTS_DIR` | `/app/artifacts/category/latest` |
| `LOG_LEVEL` | `INFO` |
| `PORT` | `8000` (Railway may inject `PORT` — map uvicorn if needed) |

If Railway sets `PORT`, override start command:

```bash
uvicorn intent_task_ai.api.app:app --host 0.0.0.0 --port $PORT
```

4. **Ensure artifacts are in the image**  
   Dockerfile `COPY artifacts ./artifacts`. Train locally before board commit so `latest/` is present.

5. **Health check**  
   Path: `/health` — expect `{"status":"ok",...}`.

6. **Smoke after deploy**

```bash
export BASE=https://<your-app>.up.railway.app
curl -s "$BASE/health"
curl -s "$BASE/v1/parse-task" -H 'Content-Type: application/json' \
  -d '{"text":"Mai 9h họp sprint, ưu tiên cao","locale":"vi-VN"}'
curl -s "$BASE/metrics"
```

## Monitoring

- **Logs:** Railway log stream — each request emits structured `http_request … latency_ms=…`
- **Metrics:** scrape `GET /metrics` (Prometheus text) for `parse_task_total` and latency sum
- **Rollback:** redeploy previous Railway deployment; artifacts are versioned under `artifacts/category/<version>/`

## Out of scope

Auth, custom domain TLS beyond Railway defaults, GPU workers, multi-region.
