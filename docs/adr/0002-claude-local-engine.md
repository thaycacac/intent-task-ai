# ADR 0002: Optional Claude (local CLI) parse engine

- **Status:** Accepted
- **Date:** 2026-10-09
- **Context:** ADR 0001 rejected an LLM as the *core* parser but left an LLM as an "optional later baseline". We now want to compare the hybrid pipeline with Claude, and use it interactively from the UI.

## Decision

Add `engine: "hybrid" | "claude"` to `POST /v1/parse-task` (default `hybrid`, additive). `claude` shells out to the locally installed Claude Code CLI (`claude -p`, tools disabled, `--json-schema`), reusing the developer's existing login — no API key handling in this service.

## Consequences

- Hybrid stays the production path; contract for existing clients is unchanged.
- `claude` is for local use / baseline comparison: slower (seconds), non-deterministic, needs the CLI. When unavailable the API returns 503 and `GET /v1/engines` reports `claude: false` so the UI disables it.
- Output values are validated against the same enums; invalid values fall back to `other` / `medium` / `null`.
- Not suitable for the Docker deploy as-is; a hosted variant would use the Anthropic API with a key (a separate decision).
