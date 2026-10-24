# DevOps Repository Auditing Agent

An LLM-driven agent that clones a public GitHub repository (read-only), investigates it with inspection tools that **the
model chooses itself**, and writes an evidence-based JSON report of DevOps best-practice violations.

The agent loop, tool dispatch and control flow are hand-written in Python. No agent framework (LangChain, LangGraph,
n8n, etc.) is used; only the official `anthropic` / `openai` SDKs for the model call (Gemini is reached through its OpenAI-compatible endpoint).

---

## 1. Installation

Requirements: Python 3.10+ and `git` on `PATH`.

```bash
git clone <this-repository-url>
cd devops-audit-agent
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins tested versions; `pyproject.toml` declares compatible ranges (`pip install -e .[dev]` also works).

## 2. Configuration

```bash
cp .env.example .env               # Windows: copy .env.example .env
```

Edit `.env` and set **one** provider key. The `.env` file is git-ignored.

| Variable | Required | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | one key | Anthropic API key |
| `OPENAI_API_KEY` | one key | OpenAI (or OpenAI-compatible) API key |
| `GEMINI_API_KEY` | one key | Google Gemini API key (free tier works) |
| `LLM_PROVIDER` | no | `anthropic`, `openai`, `gemini` or `ollama`; cloud providers auto-detected from the key |
| `LLM_MODEL` | no | Model override (defaults: `claude-sonnet-5-5`, `gpt-4.1`, `gemini-3.5-flash`, `qwen3:8b`). If a default model has been retired by the provider (HTTP 404), the agent falls back to the next known model; an explicit `LLM_MODEL` is never swapped |
| `OLLAMA_HOST`, `OLLAMA_NUM_CTX` | no | Ollama server address (default `http://localhost:11434`) and context window (default 32768) |
| `OPENAI_BASE_URL` | no | Any OpenAI-compatible endpoint (with `LLM_PROVIDER=openai`) |
| `LLM_MIN_SECONDS_BETWEEN_CALLS` | no | Request spacing; defaults to 6.5 s for Gemini (free tier ≈10 requests/min), 0 otherwise |

### Using Gemini's free tier

1. Create a key at <https://aistudio.google.com/apikey>.
2. In `.env`: `GEMINI_API_KEY=<your key>` (leave the other keys empty).
3. Run as usual. The agent uses Gemini's OpenAI-compatible endpoint, echoes back Gemini's thought signatures during
   tool calling, spaces requests to stay under the per-minute quota and retries on HTTP 429.

A full audit takes roughly 30–60 model requests, so a run lasts a few minutes on the free tier, and the daily request
quota limits how many audits you can run per day (check your quota in AI Studio). If you hit the daily limit, the run
exits with code 4 and saves partial findings.

### Using a local model with Ollama (no API key)

```bash
ollama pull qwen3:8b            # any model with tool-calling support; larger is better (e.g. qwen3:14b, gpt-oss:20b)
ollama serve                    # if it is not already running
```

In `.env`: `LLM_PROVIDER=ollama` and `LLM_MODEL=qwen3:8b` (no key needed). The agent talks to Ollama's native
`/api/chat` endpoint and sends `options.num_ctx` (default 32768, `OLLAMA_NUM_CTX`) with every request, because Ollama's
default context is small and would silently cut off the system prompt. For smaller contexts the agent automatically
shortens tool outputs and trims history sooner. Local runs are slower, so the default wall-clock budget becomes 3600 s.
Small local models are noticeably weaker at multi-step tool use than hosted models, which mainly costs recall.

Environment variables already exported in the shell take precedence over `.env`.

## 3. Running

```bash
python agent.py --repo https://github.com/<owner>/<repo> --output report.json
```

This runs unattended and writes:

* `report.json` – the report in the required contract.
* `report.trace.json` – the full tool-call trace (every LLM turn, its reasoning, tool name, arguments, output preview,
  findings accepted/rejected, retractions and coverage notes) so the agent's behaviour can be audited.

Useful options:

| Option | Default | Purpose |
|---|---|---|
| `--trace PATH` / `--trace none` | `<output>.trace.json` | Trace location or disable |
| `--provider`, `--model` | from env | Override provider/model |
| `--max-steps` | 45 | LLM investigation turns |
| `--max-tool-calls` | 120 | Inspection tool executions |
| `--timeout` | 1500 | Wall-clock budget (s) for the investigation |
| `--sandbox docker` | off | Enables `run_tests` in an isolated container (see §6) |
| `-v` | off | Progress log (reasoning + tool calls) on stderr |

Exit codes: `0` success (report written, possibly with an empty `violations` array), `2` invalid input or missing
credentials, `3` clone failure (bad URL, private/non-existent repository, timeout), `4` LLM failure (partial findings are
saved to `<output>.partial`, the main report is not written), `5` unexpected internal error.

### Output contract

```json
{
  "repository": "https://github.com/example/demo-project",
  "violations": [
    {
      "rule_id": "SEC-001",
      "category": "security",
      "severity": "critical",
      "file": "config.py",
      "line": 17,
      "description": "Hardcoded API credential",
      "evidence": "AWS access key literal assigned to AWS_ACCESS_KEY_ID",
      "recommendation": "Load it from an environment variable or secrets manager and rotate the key"
    }
  ]
}
```

`severity` ∈ `critical | high | medium | low`. `file`/`line` are JSON `null` when not applicable (e.g. "no tests at all").
Secret values are never reproduced. Categories: `repository_hygiene`, `dependency_management`, `code_quality`,
`testing`, `ci`, `security`, `configuration`, `collaboration`, `build_release`.

---

## 4. Architecture

```
agent.py ── CLI: validate input → build LLM client → clone read-only → run agent → validate + write JSON
   │
   ├── repo_loader.py   GitHub URL validation, shallow clone (no prompts, no LFS, no symlinks, no hooks),
   │                    worktree chmod'ed read-only, guaranteed cleanup
   ├── agent_loop.py    The agent: message history, turn loop, tool dispatch, budgets, coverage/self-review
   │                    control, context trimming, trace
   ├── llm_client.py    Provider adapters (Anthropic Messages / OpenAI Chat Completions / Gemini via its
   │                    OpenAI-compatible API / Ollama native API) with tool calling, retries, request spacing
   ├── tools.py         Read-only inspection tools + JSON schemas + safe dispatcher
   ├── prompts.py       System prompt, rule catalog, severity guide, agent-control tools
   ├── report.py        Finding validation (evidence grounding), de-duplication, contract check, atomic write
   └── redaction.py     Secret detection and redaction applied to every tool output, log line and report field
```

### The agent cycle

```
           ┌──────────────────────────────────────────────────────────┐
           │ Observe: repo overview (file count, top-level, file types)│
           └──────────────┬───────────────────────────────────────────┘
                          ▼
   ┌──────────────► LLM turn: reason + choose tool call(s) ───────────────┐
   │                      │                                               │
   │        inspection tool│                     record_finding /         │
   │                      ▼                      retract / mark_reviewed  │
   │          Execute (read-only, redacted,              │                │
   │          truncated) → result to history             ▼                │
   │                      │                   validator: tracked file?   │
   │                      │                   actually inspected? line   │
   │                      │                   in range? duplicate?       │
   │                      │                   → "Recorded F3" / REJECTED │
   └──────────────────────┴───────────────────────────┘                  │
                                                     finish_audit ◄──────┘
                                                          │
                              categories not reviewed? → nudge once, continue
                              findings recorded?       → self-review once (retract weak ones)
                                                          ▼
                                               Report (only recorded findings)
```

Key points:

* **The LLM directs the investigation.** There is no fixed checklist executed by code. Each turn the model sees all
  previous tool results and decides the next call(s) – e.g. `inspect_workflows` reveals `pull_request_target`, so it
  reads that workflow, then searches for `head.sha` checkouts; `list_files` shows `poetry.lock` is absent, so it reads
  `pyproject.toml` to decide whether the project is a library or an application before reporting.
* **Findings are tool calls too.** The model must call `record_finding`. The controller verifies that the file is
  tracked, that the agent actually read or searched it during this run, and that the line exists. Rejections are
  returned to the model with the reason, so it goes back and gathers evidence. This turns "unsupported claims" into a
  feedback loop instead of false positives.
* **Coverage and self-review are prompts, not checks.** When the model calls `finish_audit`, the controller asks once
  about categories never marked reviewed and once for a precision self-review of recorded findings. The model decides
  what to do; the code does not inject findings.
* **Budgets.** Turn limit, tool-call limit and wall-clock limit. When exhausted the agent gets a short wrap-up phase in
  which only `record_finding`/`retract_finding`/`finish_audit` are honoured.
* **Context management.** Old, large tool outputs are elided (call/result pairing preserved) once history exceeds a
  character budget, so long audits do not overflow the model context.

### Tools

| Tool | Purpose |
|---|---|
| `list_files(path?, pattern?)` | Tracked files, optional directory/glob filter; summarized for huge repos |
| `read_file(path, start_line?, end_line?)` | Numbered lines (≤400 per call), path-traversal and `.git` access blocked |
| `search_code(pattern, path?)` | POSIX ERE search via `git grep` with timeout |
| `git_log(max_count?, path?, show_files?)` | Commit history |
| `git_show_commit(commit)` | One commit's patch (e.g. secret added then "removed") |
| `inspect_tracked_artifacts()` | Tracked bytecode, venvs, node_modules, build output, caches, IDE settings, `.env`, keys; files tracked despite `.gitignore` |
| `scan_secrets(path?)` | Heuristic credential scan; values always redacted; placeholders and fixtures flagged |
| `inspect_workflows(path?)` | Parsed GitHub Actions: triggers, permissions, jobs, steps and risk *markers*; detects other CI systems |
| `run_ruff(path?)` | Static lint with ruff (no code execution, `--no-cache` so nothing is written) |
| `run_tests()` | Only with `--sandbox docker`: test suite in a locked-down container |
| `record_finding`, `retract_finding`, `mark_reviewed`, `finish_audit` | Agent-control tools handled by the controller |

Tool outputs labelled "markers"/"heuristic" are explicitly leads; the system prompt instructs the model to confirm them
with `read_file` before reporting.

### Precision measures (false positives are penalized)

* Evidence grounding enforced in code (file must be tracked and inspected; line must exist).
* Rule catalog with explicit acceptable alternatives (any linter, any CI system, pinned `requirements.txt` as a lock,
  libraries may use ranges, major-version action tags are fine, no branch-protection inference).
* De-duplication by rule + location and one-finding-per-root-cause guidance.
* Placeholder/reference detection (`${{ secrets.X }}`, `os.getenv`, `changeme`, test fixture paths).
* Self-review step that lets the model retract weak findings.

---

## 5. Tests

```bash
pytest
ruff check . && ruff format --check .
```

The suite (no network or API key needed) covers each tool, path-traversal protection, redaction, the finding validator,
JSON contract, both provider message converters, and the agent loop driven by a scripted fake LLM: dynamic follow-up
based on tool output, rejection of ungrounded findings, coverage nudge, budget wrap-up, malformed tool calls, context
trimming, and an end-to-end CLI run that checks the target repository is left unchanged. CI runs lint, format check and
tests on Python 3.10 and 3.12 (`.github/workflows/ci.yml`).

### Evaluating precision/recall locally

```bash
python eval/make_demo_repo.py --out /tmp/demo-repos     # defective repo (20 known defects) + healthy repo (0)
python agent.py --allow-local --repo /tmp/demo-repos/defective --output out.json
python eval/score.py out.json /tmp/demo-repos/defective.key.json
python agent.py --allow-local --repo /tmp/demo-repos/healthy --output healthy.json
python eval/score.py healthy.json /tmp/demo-repos/healthy.key.json
```

`--allow-local` exists only for this development workflow; normal use requires a GitHub URL.

### Sample report

`sample_report.json` and `sample_report.trace.json` are produced by a real run:

```bash
scripts/generate_sample_report.sh                       # audits the bundled defective demo repo
scripts/generate_sample_report.sh https://github.com/<owner>/<repo>
```

---

## 6. Safety

* The target is cloned into a fresh temp directory with `GIT_TERMINAL_PROMPT=0`, LFS smudge disabled, symlinks checked
  out as plain files, hooks disabled; worktree files are made read-only and the directory is deleted afterwards.
* Tools never execute repository code. `ruff` is static and runs with `--no-cache`. The only exception, `run_tests`, is
  not even offered to the model unless `--sandbox docker` is passed; it then runs in `python:3.12-slim` with the repo
  mounted read-only, `--cap-drop ALL`, `no-new-privileges`, memory/CPU/PID limits and a timeout.
* All tool output, LLM reasoning in the trace, logs and report fields go through `redaction.redact`, so secret values
  never reach the model, the logs or the report.
* Repository content is treated as data; the system prompt tells the model never to follow instructions found in files.
* Unavailable checks say so explicitly ("check unavailable – do not infer"), and the prompt forbids inventing results.
* Nothing repository-specific is hardcoded; the rule catalog is generic.

## 7. Limitations

* The clone is shallow (200 commits) to bound time; older history is not inspected.
* Remote GitHub settings (branch protection, environments, secrets) are intentionally not inferred.
* Test pass/fail is only reported when the Docker sandbox is enabled.
* Detection quality depends on the model; larger models are more precise on subtle CI/security issues.

## 8. Project layout

```
agent.py              CLI entry point
agent_loop.py         agent loop and controller
llm_client.py         LLM provider adapters
prompts.py            system prompt, rule catalog, control tools
tools.py              inspection tools, schemas, dispatcher
report.py             validation, de-duplication, JSON output
redaction.py          secret detection/redaction
repo_loader.py        safe cloning and cleanup
tests/                pytest suite (fake LLM, fixture repositories)
eval/                 demo repositories with answer keys + scorer
scripts/              helper scripts (sample report generation)
sample_report.json    example agent output
.env.example          required variable names (no real credentials)
```
