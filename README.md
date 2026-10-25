# DevOps Repository Auditing Agent

An LLM-driven agent that clones a public GitHub repository (read-only), investigates it with inspection tools that **the
model chooses itself**, and writes an evidence-based JSON report of DevOps best-practice violations.

Assignment 2 of *MLOps & Production AI Engineering*. Repository: <https://github.com/abdullahhqayyum/devops-repository-auditing-agent>

* The agent loop, tool dispatch and control flow are hand-written Python (`agent_loop.py`). No agent framework
  (LangChain, LangGraph, n8n, ...) is used; the model call goes through plain provider SDKs (`anthropic`, `openai`)
  or plain HTTP (Ollama). Gemini and OpenRouter are reached through their OpenAI-compatible endpoints.
* The model decides which tool to call next from what it has seen. There is no fixed checklist and no static scanner whose
  output is merely summarised.
* The audited repository is never modified and its code is never executed.

---

## 1. Requirements and installation

Requirements: **Python 3.10 or newer** and **`git` on `PATH`**. Dependencies are pinned in `requirements.txt`
(`anthropic`, `openai`, `PyYAML`, `python-dotenv`, `ruff`, `pytest`); `pyproject.toml` declares compatible ranges.

```bash
git clone https://github.com/abdullahhqayyum/devops-repository-auditing-agent
cd devops-repository-auditing-agent
python -m venv .venv
pip install -r requirements.txt     # after activating the environment, see below
```

Activate the virtual environment before installing:

| Shell | Command |
|---|---|
| macOS / Linux | `source .venv/bin/activate` |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` (once per session, if scripts are blocked: `Set-ExecutionPolicy -Scope Process RemoteSigned`) |
| Windows cmd | `.venv\Scripts\activate.bat` |

Activation is convenient but not required: the agent finds `ruff` through the Python interpreter that runs it.

## 2. Configuration (API credentials)

```bash
cp .env.example .env                # Windows: copy .env.example .env
```

Edit `.env` and set **one** provider key. `.env` is git-ignored; never commit it. Variables already exported in your shell
take precedence over `.env`.

| Variable | Meaning |
|---|---|
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) / `OPENROUTER_API_KEY` | Credentials for the provider you use (one is enough) |
| `LLM_PROVIDER` | `anthropic`, `openai`, `gemini`, `openrouter` or `ollama`. Optional for cloud providers (detected from the key) |
| `LLM_MODEL` | Optional model override |
| `OPENAI_BASE_URL` | Optional OpenAI-compatible endpoint (used with `LLM_PROVIDER=openai`) |
| `LLM_MIN_SECONDS_BETWEEN_CALLS` | Minimum gap between model requests. Default 6.5 s for Gemini (free-tier friendly), 3.5 s for OpenRouter, 0 otherwise. Raise it (for example 13) if you see per-minute rate limits |
| `OLLAMA_HOST`, `OLLAMA_NUM_CTX` | Local Ollama server (default `http://localhost:11434`) and context window (default 32768) |

### Provider and model selection

1. **Provider**: `--provider`, else `LLM_PROVIDER`, else detected from the keys present (Anthropic, then OpenAI, then Gemini,
   then OpenRouter, when several are set). Ollama must be selected explicitly.
2. **Model**: `--model`, else `LLM_MODEL`, else the provider default below. Only a *default* model falls back to the next
   known name when the provider reports it missing (HTTP 404); an explicitly chosen model is never swapped.

| Provider | Default model | Notes |
|---|---|---|
| Gemini | `gemini-3.6-flash` | OpenAI-compatible endpoint; **the only provider exercised live so far** |
| Anthropic | `claude-sonnet-5-5` | Messages API with native tool use (covered by mocked tests only) |
| OpenAI | `gpt-4.1` | Chat Completions; also any OpenAI-compatible server (covered by mocked tests only) |
| OpenRouter | `openrouter/free` | One key, many models, through the OpenAI-compatible adapter. Needs a **tool-capable** model; see below (mocked tests; one live smoke attempt only partly succeeded, see below) |
| Ollama | `qwen3:8b` | Local, no key. Needs `ollama serve` and a tool-calling model you pulled |

The grade does not depend on the provider; any one of them works with the same command.

### Using OpenRouter

OpenRouter gives one API key access to many hosted models. It is a first-class provider (`openrouter`) that reuses the
OpenAI-compatible adapter, the same agent loop and the same retry, redaction and trace code as the other providers.

```bash
# .env (never commit it)
OPENROUTER_API_KEY=<your key from https://openrouter.ai/keys>
LLM_PROVIDER=openrouter
LLM_MODEL=openrouter/free          # or a pinned slug, see below

python agent.py --repo https://github.com/<owner>/<repo> --output report.json --provider openrouter --model openrouter/free
```

* **Credentials and endpoint.** The key is read from `OPENROUTER_API_KEY` (never from `OPENAI_API_KEY`); the endpoint is
  `https://openrouter.ai/api/v1`.
* **Models.** `--model` or `LLM_MODEL` takes an OpenRouter slug such as `nvidia/nemotron-3-super-120b-a12b:free`. Slugs are
  passed through verbatim and are never swapped for another model, because a silent swap could land on a paid one.
* **The agent requires tool calling.** A model without it fails with a clear message (OpenRouter answers
  "No endpoints found that support tool use"); pick a model whose `supported_parameters` include `tools`
  (<https://openrouter.ai/api/v1/models> lists them).
* **`openrouter/free` versus a pinned model.** `openrouter/free` routes among free models that support the features of the
  request, so it may pick a different model on each call. That is convenient for a smoke test but unsuitable for comparing
  detection quality. For reproducible runs pin one slug. When this was written, the public catalog listed these free,
  tool-capable models with a 256k context: `nvidia/nemotron-3-super-120b-a12b:free`, `google/gemma-4-31b-it:free` and
  `cohere/north-mini-code:free`. Free models can be withdrawn, so check the catalog; whether any of them performs well on this
  agent has **not been measured here**.
* **Free-tier limits and cost.** Free (`:free`) models are documented at 20 requests per minute and 50 per day for accounts
  with under 10 credits purchased, shared across all free models (confirm on OpenRouter's limits page). A negative credit
  balance can return HTTP 402 even for free models. Paid models are billed per token; the agent never selects one unless you
  name it.
* **Behaviour on errors.** 402 (insufficient credits), 401 and 404 are never retried and explain what to do. A daily
  free-model limit stops the run at once and prints the reset time when OpenRouter reports one; a per-minute limit is retried
  a bounded number of times. A completion that arrives without `choices` is handled like the error it carries.
* **Context budget.** The agent trims old tool output sooner for OpenRouter (about 100k characters of history) because the
  models behind the router have different context windows.
* **Checking your free quota.** `GET https://openrouter.ai/api/v1/key` with your key as a bearer token reports spend and
  `free_model_daily_requests` (`used`, `limit`, `remaining`).
* **Live status.** One real run against `octocat/Hello-World` with `google/gemma-4-31b-it:free` authenticated, was accepted
  with our tool schemas, and the model chose `list_files` by itself, but the run did **not complete**: OpenRouter then
  answered HTTP 429 "temporarily rate-limited upstream" for that model, and the agent stopped after its bounded retries
  (exit 4, labeled partial output). That limit is the upstream provider's shared free capacity for one model, separate from
  your account quota (the account showed 0 of 50 free requests used and no spend). A pinned free model can therefore be
  throttled for a while; `openrouter/free` routes among several free models and may avoid that, at the cost of
  reproducibility. The run was not repeated, so full-audit behaviour on OpenRouter is still unverified.

## 3. Running

```bash
python agent.py --repo https://github.com/<owner>/<repo> --output report.json
```

It runs unattended, with no prompts, and writes `report.json` plus `report.trace.json` (every model turn, tool call,
accepted or rejected finding, and why the run stopped).

| Option | Default | Purpose |
|---|---|---|
| `--trace PATH` / `--trace none` | `<output>.trace.json` | Trace location, or disable |
| `--provider`, `--model` | from env | Override provider and model |
| `--max-steps` | 45 | Model turns for the investigation (a short wrap-up follows) |
| `--max-tool-calls` | 120 | Inspection tool executions (a short wrap-up follows) |
| `--timeout` | 1500 (3600 for Ollama) | Wall-clock budget in seconds. Running out ends the audit as **incomplete** |
| `--sandbox docker` | off | Offers the `run_tests` tool, run inside a locked-down container |
| `--allow-local` | off | Accept a local git path as `--repo` (evaluation and development only) |
| `-v` | off | Progress log on stderr |

### Exit codes

| Code | Meaning | Output |
|---|---|---|
| 0 | Investigation completed | `report.json` (valid JSON, `violations` may be empty) |
| 2 | Invalid input, missing credentials, or bad options | none |
| 3 | Repository could not be cloned (bad URL, private or missing repository, timeout) | none |
| 4 | **Incomplete investigation**: LLM failure, exhausted quota, or the wall-clock budget ran out | `<output>.partial` with the findings so far, labeled `"incomplete": true` and a `stop_reason`. `report.json` is **not** written, so a partial run is never presented as a finished audit |
| 5 | Unexpected internal error | none |
| 130 | Interrupted | none |

The trace of an incomplete run records `"status": "incomplete"` and the stop reason.

## 4. Output

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
      "evidence": "Credential assigned directly in source code",
      "recommendation": "Use a secrets manager or environment variable"
    }
  ]
}
```

`severity` is one of `critical`, `high`, `medium`, `low`. `file` and `line` are JSON `null` when not applicable (for
example "no tests at all"). Secret values are never reproduced. Every report is validated against this contract before it
is written (`report.validate_report`).

`sample_report.json` is a real output of the agent: a completed audit of `https://github.com/octocat/Hello-World`
(a one-file repository, so the violations list is genuinely empty). `sample_report.trace.json` is that run's trace. It was
recorded before the Ruff-detection fix described below, so its `run_ruff` call honestly reports Ruff as unavailable.

## 5. Architecture

```
agent.py          CLI: validate input, build the LLM client, clone read-only, run the agent, validate and write JSON
agent_loop.py     The agent: message history, turn loop, tool dispatch, budgets, coverage and self-review nudges, trace
llm_client.py     Provider adapters (Anthropic, OpenAI-compatible incl. Gemini and OpenRouter, Ollama), retries, quota handling, deadline
tools.py          Read-only inspection tools, their JSON schemas, argument validation and dispatch
prompts.py        System prompt, rule catalog, severity guide, agent-control tools
report.py         Finding validation (evidence grounding), de-duplication, contract check, atomic write
redaction.py      Secret detection and redaction applied to tool output, logs, traces and report text
repo_loader.py    GitHub URL validation, hardened shallow clone, read-only checkout, cleanup
```

### The agent cycle

Observe, reason, select tool, execute, interpret, record:

1. The agent shows the model a short overview of the repository (file count, top-level entries, file types).
2. Each turn the model sees all earlier tool results and returns one or more tool calls. Independent calls are batched in
   one turn; dependent ones are issued after the result they depend on arrives.
3. The controller executes each call, redacts and truncates the result, and feeds it back.
4. Findings are tool calls too (`record_finding`). The controller checks that the file is tracked, that the agent actually
   read or searched it during this run, and that the line exists. A rejection is returned to the model with the reason, so
   it gathers evidence and tries again. The model can retract findings (`retract_finding`).
5. `mark_reviewed` documents which of the nine categories were examined. When the model calls `finish_audit` the controller
   asks once about unreviewed categories and once for a precision self-review of recorded findings; the model decides what
   to do and the code never injects findings.
6. Budgets: the step and tool-call limits trigger a short wrap-up in which only `record_finding`, `retract_finding` and
   `finish_audit` are honoured, then a normal finish. The wall-clock limit is stricter: it ends the audit as incomplete
   (exit 4) with no wrap-up.

### Inspection tools

| Tool | Purpose |
|---|---|
| `list_files(path?, pattern?)` | Discover tracked files; summarised for huge repositories |
| `read_file(path, start_line?, end_line?)` | Numbered lines (at most 400 per call); path traversal and `.git` access are blocked |
| `search_code(pattern, path?)` | POSIX extended-regex search over tracked text files (`git grep`) |
| `git_log(max_count?, path?, show_files?)` | Commit history (the clone is shallow, depth 200) |
| `git_show_commit(commit)` | One commit's patch, for example a secret added and later removed |
| `inspect_tracked_artifacts()` | Tracked bytecode, virtualenvs, build output, caches, IDE settings, `.env`, keys; files tracked despite `.gitignore` |
| `scan_secrets(path?)` | Heuristic credential scan; values always redacted, placeholders and fixtures flagged |
| `inspect_workflows(path?)` | Parsed GitHub Actions workflows: triggers, permissions, jobs, steps and risk *markers*; other CI systems |
| `run_ruff(path?)` | Static Ruff lint (no code execution) |
| `run_tests()` | Only with `--sandbox docker`: the test suite in a container |

Tool output labelled "markers" or "heuristic" is a lead, not a conclusion; the prompt tells the model to confirm it with
`read_file`. Malformed tool arguments produce an `ERROR:` result the model can correct, never a crash.

### Audit categories

Repository hygiene, dependency management, code quality, testing, CI (GitHub Actions), security, configuration,
collaboration, build and release. `prompts.py` holds the rule catalog (for example `HYG-003` tracked generated files,
`DEP-003` inconsistent specifications, `CI-003` ineffective gates, `SEC-004` dangerous privileged triggers) together with
the acceptable alternatives that must not be reported (any linter, any CI system, pinned `requirements.txt` as a lock,
libraries may use ranges, no inference about remote branch protection).

## 6. Retry, quota and time-budget handling

* The SDK clients are created with `max_retries=0`; `llm_client._retrying` is the only retry layer.
* **HTTP 429** gets at most 3 attempts. A server hint (`Retry-After`, `retry-after-ms`, a `RetryInfo` delay, or "retry in N s"
  text) is honoured up to 60 s; a longer one ends the run. Without a hint the wait is exponential (5 s, 10 s).
* A 429 that identifies a **daily** quota window (`quotaId` containing `PerDay` or `Daily`, wording that names a per-day
  limit) or OpenAI's `insufficient_quota` stops at once: waiting seconds cannot clear it. Generic "check your billing"
  text is not treated as exhaustion, because per-minute limits use it too.
* **Timeouts, connection errors and 5xx** get up to 6 attempts with exponential backoff (3, 6, 12, 24, 48 s). Ollama keeps
  its own 3 attempts.
* The agent gives the adapter an absolute deadline derived from `--timeout`. Retry sleeps, request spacing and each request's
  timeout respect it. SDK timeouts apply per phase (connect, read, write) rather than to the whole request, so a stalled
  network can still overrun the deadline slightly.
* Error text is redacted before it is logged or stored, and any configured API key is scrubbed from it.

## 7. Safety

* The target is cloned into a fresh temporary directory with prompts disabled, no LFS download, symlinks as plain files and
  hooks disabled. The worktree is made read-only and deleted afterwards. `--allow-local` copies a local repository the same way.
* Tools never execute repository code. Ruff runs as `python -I -m ruff` (isolated mode), so a `ruff.py` inside the audited
  repository cannot shadow the real package. `run_tests` is not offered unless `--sandbox docker` is passed; it then runs
  with the repository mounted read-only, all capabilities dropped and resource limits.
* Every string that leaves a tool, plus log lines, the trace and the report text, passes through `redaction.redact`: known
  token formats, URL passwords, quoted `NAME = "value"` assignments, unquoted `.env` and INI assignments (redacted before
  line numbers or diff markers are added), and unquoted YAML values under secret-named keys.
* Repository content is treated as data; the prompt tells the model never to follow instructions found in files.
* A check that is unavailable says so (for example "ruff is not installed ... do not infer results"); the model is told not
  to invent its result.

## 8. Tests and linting

```bash
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

The suite needs no network and no API key. It covers each tool (including path-traversal protection and malformed
arguments), redaction, the finding validator and JSON contract, the provider message converters, the OpenRouter integration, the retry, quota and
deadline logic, the agent loop driven by a scripted fake model (dynamic follow-up, batched calls, rejection of ungrounded
findings, coverage nudge, budgets, incomplete runs), the CLI end to end, and the evaluation scorer. **No test calls a real
model API.** CI (`.github/workflows/ci.yml`) runs lint, format check and tests on Python 3.10 and 3.12.

## 9. Evaluation fixtures and scoring

`eval/make_demo_repo.py` builds two local git repositories with answer keys: `defective` (20 injected defects across all
categories) and `healthy` (no known defects).

```bash
# macOS / Linux
python eval/make_demo_repo.py --out /tmp/demo-repos
python agent.py --allow-local --repo /tmp/demo-repos/defective --output /tmp/defective.json
python eval/score.py /tmp/defective.json /tmp/demo-repos/defective.key.json
python agent.py --allow-local --repo /tmp/demo-repos/healthy --output /tmp/healthy.json
python eval/score.py /tmp/healthy.json /tmp/demo-repos/healthy.key.json
```

```powershell
# Windows PowerShell
python eval/make_demo_repo.py --out "$env:TEMP\demo-repos"
python agent.py --allow-local --repo "$env:TEMP\demo-repos\defective" --output "$env:TEMP\defective.json"
python eval/score.py "$env:TEMP\defective.json" "$env:TEMP\demo-repos\defective.key.json"
```

`--allow-local` exists only for this workflow; normal use requires a GitHub URL. `make_demo_repo.py` deletes and recreates
`<out>/defective` and `<out>/healthy`, so point `--out` at a scratch directory.

### How to read the scores

A reported violation counts as a **true positive (TP)** when it matches an unmatched answer-key entry (same file and rule id;
the lenient line also accepts the same rule family). Anything else reported is a **false positive (FP)**, including a
duplicate of an already matched defect. Known defects nobody reported are **false negatives (FN)**.

* Precision = TP / (TP + FP) and recall = TP / (TP + FN); F1 is their harmonic mean.
* Undefined values print **N/A**, never a flattering number: precision is N/A when nothing was reported, recall is N/A when
  the key is empty (a healthy repository), F1 is N/A unless both are defined. On a healthy repository the FP count is the
  meaningful figure.
* A report marked `"incomplete": true` is labelled **INCOMPLETE**, gets no metrics, and `score.py` exits with status 2.
* The answer key lists the *injected* defects. An unmatched finding can still be a genuine issue, so false positives should
  be reviewed by hand before they are held against the agent.

### Results that have actually been obtained

All runs in the table used Gemini `gemini-3.6-flash` except the last row, which used an OpenRouter model. The fixtures are
the author's own and were written alongside the rule catalog, so they measure whether the machinery works, not performance
on unseen repositories. Scores from different models are not comparable with one another.

| Run | Status | Outcome |
|---|---|---|
| `octocat/Hello-World` (public) | **Complete** | 0 violations (the repository holds one README). 7 model turns, 7 inspection calls, 32,861 input and 611 output tokens. This is `sample_report.json` |
| `defective` fixture, first attempt | **Incomplete** (daily quota) | Stopped after 11 turns with no findings recorded. Not an accuracy result |
| `defective` fixture, second run | **Complete** | Scorer: TP 19, FP 1, FN 1; precision, recall and F1 0.95 (strict and lenient). 17 model turns, 22 inspection calls, 19 HTTP requests (two transient 503 responses retried), 209,936 input and 4,404 output tokens, about 290 s with 13 s request spacing |
| `healthy` fixture | **Not completed** | The daily quota was exhausted on the first request, so the false-positive rate on a healthy repository is **unmeasured** |
| `octocat/Hello-World`, OpenRouter `google/gemma-4-31b-it:free` | **Incomplete** | Authenticated, tool schemas accepted, one model-chosen `list_files` call, then upstream HTTP 429 on the next turn (6 requests in all, 2 turns). Exit 4 with a labeled partial file; not an accuracy result and not repeated |

About the one scored miss in the second `defective` run: all 20 defects were reported, but HYG-003 (a tracked `.pyc`) was
recorded with `file: null`. At that time `read_file` did not register a binary file as inspected, the evidence check rejected
every attempt to cite the `.pyc`, and the model spent about half of its turns on it before falling back to a location-less
finding. The scorer therefore counted one FP and one FN for what is one genuine defect. `read_file` and
`inspect_tracked_artifacts` now register such files, with regression tests; **that change has not been re-run against the
live model.**

## 10. Known limitations

* Only Gemini has completed live audits. Anthropic, OpenAI and Ollama are covered by mocked tests. OpenRouter is covered by
  mocked tests plus one live attempt that was cut short by an upstream rate limit, so which free OpenRouter models complete a
  full audit with the agent's tool schema is unmeasured. Free models can be throttled or withdrawn at any time.
* Detection quality has been measured once, on the author's own fixtures. Behaviour on unseen repositories, and the false-positive
  rate on a healthy repository, are not yet measured.
* Gemini's free tier has per-minute and per-day request limits. The agent stops cleanly when the daily limit is hit
  (exit 4, labeled partial output) but cannot work around it.
* Redaction is heuristic. Unquoted YAML values are redacted only under secret-named keys and only when they contain a digit
  or a symbol; plain-word passwords, multi-word values, Kubernetes `name`/`value` pairs and flow mappings are not detected.
* The wall-clock deadline cannot interrupt a request that is already in flight beyond its own phase timeouts.
* The clone is shallow (200 commits). Remote GitHub settings (branch protection, environments, secrets) are intentionally
  not inferred. Test results are reported only with `--sandbox docker`.
* On Windows, tool output shows paths with backslashes; the controller normalises them for evidence checks.
* The `run_ruff` header says "ruff default rules E4,E7,E9,F", but the installed Ruff version also reports other families
  under `--isolated`; treat the header as approximate.

## 11. Project layout

```
agent.py  agent_loop.py  llm_client.py  prompts.py  report.py  redaction.py  repo_loader.py  tools.py
tests/                  pytest suite (scripted fake model, fixture repositories, no network)
eval/                   fixture generator with answer keys, and the scorer
scripts/                helper script to regenerate the sample report
sample_report.json      authentic example output (completed audit of octocat/Hello-World)
sample_report.trace.json  trace of that run
.env.example            variable names only, no credentials
requirements.txt  pyproject.toml  .github/workflows/ci.yml
```
