# Intent Task AI

Hybrid NLP service: free-text → structured task fields for DonDon quick-create.

| Field | Source |
|-------|--------|
| `category` | Classifier trained on AmazonScience/MASSIVE (`vi-VN` + `en-US`), mapped to `work\|personal\|errand\|learning\|health\|other` |
| `deadline` | Rule / dateparser, timezone `Asia/Ho_Chi_Minh` |
| `priority` | Keyword heuristics (`low\|medium\|high\|urgent`) |
| `task_detail` | Deterministic normalizer |

Architecture decision: [docs/adr/0001-hybrid-architecture.md](docs/adr/0001-hybrid-architecture.md)  
OpenAPI: [openapi/parse-task.yaml](openapi/parse-task.yaml)

## Quick start (local)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 1) Data → train → eval (needs Hugging Face network once)
python scripts/download_massive.py
python scripts/prepare_dataset.py
python scripts/train_category.py
python scripts/eval_category.py
python scripts/eval_golden.py

# 2) Serve
export PYTHONPATH=src
export ARTIFACTS_DIR=artifacts/category/latest
uvicorn intent_task_ai.api.app:app --host 0.0.0.0 --port 8000

# 3) Smoke
python scripts/smoke_parse.py
```

### Web UI

Mở **http://127.0.0.1:8000/** sau khi chạy server: nhập câu, chọn engine, xem category / priority / deadline / confidence.
Engine: **Hybrid** (mặc định), **Claude (local)**, hoặc **So sánh** hai engine cạnh nhau. UI là một file tĩnh tại `src/intent_task_ai/api/static/index.html` (không cần build).

Ô nhập nhận cả **OKR** nhiều dòng (`Objective:` / `O1:` / `Mục tiêu:`, `KR1:` / `Key Result:` / `Kết quả then chốt:`, gạch đầu dòng…); mỗi dòng được parse riêng và hiển thị theo Objective → Key Results. API tương ứng: `POST /v1/parse-okr`.

### Gemini engine (mặc định khi có key)

Copy `.env.example` → `.env` (đã gitignore), điền `GEMINI_API_KEY`, rồi chạy server với `--env-file .env`:

```bash
PYTHONPATH=src ARTIFACTS_DIR=artifacts/category/latest uvicorn intent_task_ai.api.app:app --port 8000 --env-file .env
```

| Env | Mặc định | Ý nghĩa |
|-----|----------|---------|
| `GEMINI_API_KEY` | — | bắt buộc để bật engine `gemini` |
| `GEMINI_MODEL` | `gemini-3.1-flash-lite` | model Gemini |
| `GEMINI_TIMEOUT_S` | `30` | timeout mỗi lần gọi |
| `DEFAULT_ENGINE` | `hybrid` | engine dùng khi request không gửi `engine` (`hybrid`/`claude`/`gemini`); không khả dụng thì tự về `hybrid` |

Lưu ý: engine `gemini` gửi nội dung văn bản đến Google API.

### Claude (local) engine — tuỳ chọn

Dùng Claude Code CLI đã cài và đăng nhập trên máy (không cần `ANTHROPIC_API_KEY`):

```bash
curl -s http://127.0.0.1:8000/v1/parse-task -H 'Content-Type: application/json' \
  -d '{"text":"Mai 9h họp sprint với team","engine":"claude"}'
```

| Env | Mặc định | Ý nghĩa |
|-----|----------|---------|
| `CLAUDE_CLI_PATH` | `claude` | đường dẫn CLI |
| `CLAUDE_MODEL` | (mặc định của CLI) | ví dụ `haiku`, `sonnet` |
| `CLAUDE_TIMEOUT_S` | `60` | timeout mỗi lần gọi |

CLI chạy headless, tắt toàn bộ tool, ép JSON schema. Docker image không có CLI nên `engine=claude` trả 503 ở đó. Xem [ADR 0002](docs/adr/0002-claude-local-engine.md).

### Docker Compose

Train (or copy) artifacts into `artifacts/category/latest/` first, then:

```bash
docker compose up --build
curl -s http://127.0.0.1:8000/health
python scripts/smoke_parse.py
```

### Example parse

```bash
curl -s http://127.0.0.1:8000/v1/parse-task \
  -H 'Content-Type: application/json' \
  -d '{"text":"Mai 9h họp sprint với team, ưu tiên cao","locale":"vi-VN"}'
```

## DonDon consumer

Contract only (no monorepo integrate): see [examples/dondon_client.py](examples/dondon_client.py).

## Deploy

One-target guide: [docs/deploy-railway.md](docs/deploy-railway.md)  
Ops: `GET /health`, `GET /metrics`, structured JSON-ish access logs.

## Layout

```
openapi/          OpenAPI 3.1 contract
docs/adr/         Architecture Decision Records
src/intent_task_ai/
  api/            FastAPI app
  category/       MASSIVE mapping + sklearn model
  deadline/       VI/EN date rules
  priority/       Heuristics
  task_detail/    Normalizer
  pipeline/       Hybrid orchestration
scripts/          download / prepare / train / eval / smoke
data/mappings/    Intent → taxonomy
data/golden/      VI golden set (≥50)
artifacts/        Versioned model bundles
examples/         DonDon example client
```

## Git policy (Thaycacac)

Agents stage changes only — **no** `git commit` / `git push`. Board commits manually after review.
