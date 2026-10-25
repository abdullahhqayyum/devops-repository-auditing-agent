# DevOps Repository Auditing Agent

An LLM-driven agent that clones a public GitHub repository (read-only), investigates it with inspection tools that **the
model chooses itself**, and writes an evidence-based JSON report of DevOps best-practice violations.

Assignment 2 of *MLOps*. Repository: <https://github.com/abdullahhqayyum/devops-repository-auditing-agent>

* The agent loop, tool dispatch and control flow are hand-written Python (`auditor/agent_loop.py`). No agent framework
  (LangChain, LangGraph, n8n, ...) is used; OpenRouter is called through the OpenAI-compatible Python SDK.
* The model decides which tool to call next from what it has seen. There is no fixed checklist and no static scanner whose
  output is merely summarised.
* The audited repository is never modified and its code is never executed.

---

## 1. Requirements and installation

Requirements: **Python 3.10 or newer** and **`git` on `PATH`**. Dependencies are pinned in `requirements.txt`
(`openai`, `PyYAML`, `python-dotenv`, `ruff`, `pytest`); `pyproject.toml` declares compatible ranges.

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

Edit `.env` and set `OPENROUTER_API_KEY`. The evaluator must supply an OpenRouter key to run this version; a key for another
service will not authenticate it. `.env` is git-ignored; never commit it. Variables already exported in your shell take
precedence over `.env`.

| Variable | Meaning |
|---|---|
| `OPENROUTER_API_KEY` | Required OpenRouter credential |
| `LLM_MODEL` | Optional OpenRouter model slug; defaults to `openrouter/free` |
| `LLM_MIN_SECONDS_BETWEEN_CALLS` | Optional minimum gap between requests; defaults to 3.5 s |

`--model` overrides `LLM_MODEL`; otherwise the agent uses `openrouter/free`. The optional `--provider openrouter` flag remains
accepted for existing commands. Old provider keys and `LLM_PROVIDER` never select an API. If an inherited `LLM_PROVIDER`
names a different provider, its `LLM_MODEL` value is ignored unless you explicitly pass `--model`.

### Using OpenRouter

OpenRouter gives one API key access to many hosted models. The client uses its OpenAI-compatible endpoint; the investigation
loop, tool dispatch, retry handling and trace remain hand-written in this repository.

```bash
# .env (never commit it)
OPENROUTER_API_KEY=<your key from https://openrouter.ai/keys>

python agent.py --repo https://github.com/<owner>/<repo> --output report.json
# Optional: --model cohere/north-mini-code:free
```

* **Credentials and endpoint.** The key is read from `OPENROUTER_API_KEY`; the endpoint is
  `https://openrouter.ai/api/v1`.
* **Models.** `--model` or `LLM_MODEL` takes an OpenRouter slug such as `cohere/north-mini-code:free`. Slugs are
  passed through verbatim and are never swapped for another model, because a silent swap could land on a paid one.
* **The agent requires tool calling.** A model without it fails with a clear message (OpenRouter answers
  "No endpoints found that support tool use"); pick a model whose `supported_parameters` include `tools`
  (<https://openrouter.ai/api/v1/models> lists them).
* **`openrouter/free` versus a pinned model.** `openrouter/free` routes among free models that support the features of the
  request, so it may pick a different model on each call. Pin a current tool-capable slug for reproducible runs. Free models
  can be withdrawn, so check the catalog before using a pinned slug.
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
* **Live evaluations.** Earlier Gemini and OpenRouter runs, including the incomplete pinned-model smoke test, are historical
  results from before this cleanup. Their outcomes and limitations are recorded in Section 9.

## 3. Running

```bash
python agent.py --repo https://github.com/<owner>/<repo> --output report.json
```

It runs unattended, with no prompts, and writes `report.json` plus `report.trace.json` (every model turn, tool call,
accepted or rejected finding, and why the run stopped).

| Option | Default | Purpose |
|---|---|---|
| `--trace PATH` / `--trace none` | `<output>.trace.json` | Trace location, or disable |
| `--provider openrouter` | OpenRouter | Optional compatibility flag; other providers are unsupported |
| `--model SLUG` | `LLM_MODEL` or `openrouter/free` | Choose an OpenRouter model |
| `--max-steps` | 45 | Model turns for the investigation (a short wrap-up follows) |
| `--max-tool-calls` | 120 | Inspection tool executions (a short wrap-up follows) |
| `--timeout` | 1500 | Wall-clock budget in seconds. Running out ends the audit as **incomplete** |
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

`sample_report.json` is an authentic historical output from a Gemini `gemini-3.6-flash` audit of
`https://github.com/octocat/Hello-World` (a one-file repository, so the violations list is genuinely empty).
`docs/sample_report.trace.json` is that run's trace. Neither artifact was regenerated for the OpenRouter-only refactor. They
predate the Ruff-detection fix described below, so the trace's `run_ruff` call honestly reports Ruff as unavailable.

## 5. Architecture

```
agent.py          CLI: validate input, build the LLM client, clone read-only, run the agent, validate and write JSON
tools.py          Read-only inspection tools, their JSON schemas, argument validation and dispatch
auditor/          Internal package containing the agent loop, OpenRouter client, prompts, reporting, redaction and repository loading
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
collaboration, build and release. `auditor/prompts.py` holds the rule catalog (for example `HYG-003` tracked generated files,
`DEP-003` inconsistent specifications, `CI-003` ineffective gates, `SEC-004` dangerous privileged triggers) together with
the acceptable alternatives that must not be reported (any linter, any CI system, pinned `requirements.txt` as a lock,
libraries may use ranges, no inference about remote branch protection).

## 6. Retry, quota and time-budget handling

* The OpenAI-compatible SDK client is created with `max_retries=0`; `llm_client._retrying` is the only retry layer.
* **HTTP 429** gets at most 3 attempts. A server hint (`Retry-After`, `retry-after-ms`, a `RetryInfo` delay, or "retry in N s"
  text) is honoured up to 60 s; a longer one ends the run. Without a hint the wait is exponential (5 s, 10 s).
* A 429 that identifies a **daily** quota window (`quotaId` containing `PerDay` or `Daily`, wording that names a per-day
  limit) or a structured `insufficient_quota` code stops at once: waiting seconds cannot clear it. Generic "check your billing"
  text is not treated as exhaustion, because per-minute limits use it too.
* **Timeouts, connection errors and 5xx** get up to 6 attempts with exponential backoff (3, 6, 12, 24, 48 s).
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

The suite needs no network or real API key. It covers each tool (including path-traversal protection and malformed
arguments), redaction, the finding validator and JSON contract, OpenRouter message conversion, the retry, quota and
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

These runs predate the OpenRouter-only refactor. The first four used Gemini `gemini-3.6-flash`; the last two used OpenRouter. The fixtures are
the author's own and were written alongside the rule catalog, so they measure whether the machinery works, not performance
on unseen repositories. Scores from different models are not comparable with one another.

| Run | Status | Outcome |
|---|---|---|
| `octocat/Hello-World` (public) | **Complete** | 0 violations (the repository holds one README). 7 model turns, 7 inspection calls, 32,861 input and 611 output tokens. This is `sample_report.json` |
| `defective` fixture, first attempt | **Incomplete** (daily quota) | Stopped after 11 turns with no findings recorded. Not an accuracy result |
| `defective` fixture, second run | **Complete** | Scorer: TP 19, FP 1, FN 1; precision, recall and F1 0.95 (strict and lenient). 17 model turns, 22 inspection calls, 19 HTTP requests (two transient 503 responses retried), 209,936 input and 4,404 output tokens, about 290 s with 13 s request spacing |
| `healthy` fixture, Gemini | **Incomplete** | The daily quota was exhausted on the first request; no Gemini healthy-fixture accuracy result |
| `octocat/Hello-World`, OpenRouter `google/gemma-4-31b-it:free` | **Incomplete** | Authenticated, tool schemas accepted, one model-chosen `list_files` call, then upstream HTTP 429 on the next turn (6 requests in all, 2 turns). Exit 4 with a labeled partial file; not an accuracy result and not repeated |
| `healthy` fixture, OpenRouter `openrouter/free` | **Complete** | 0 findings and 0 verified false positives on the eight-file fixture; 6 model turns and 11 inspection calls. Strict and lenient scorer: TP=0, FP=0, FN=0; precision, recall and F1 are N/A |

The completed OpenRouter healthy run took 106 seconds and used 31,770 input plus 3,082 output tokens. There were seven
completion attempts: six HTTP 200 responses and one connection failure recovered by the normal per-request retry; no
upstream rate limit occurred. The response `model` fields identified six different free models:
`cohere/north-mini-code:free`, `recursive/flint-v1.0:free`, `nvidia/nemotron-3-ultra-550b-a55b:free`,
`liquid/lfm-2.5-2.6b:free`, `poolside/laguna-s-2.1:free`, and `nvidia/nemotron-3.5-super-vl:free`. The router may choose
different models on another run, so this is a check of the free-router workflow, not a result for one pinned model. The
agent read all eight tracked files, checked tracked artifacts, ran Ruff, marked all nine categories reviewed and called
`finish_audit`. It did not call `scan_secrets` or run tests in a sandbox; a security review note inaccurately says
`scan_secrets` ran. Zero reported findings means zero false positives on this fixture, but does not establish a general
false-positive rate on unseen repositories.

About the one scored miss in the second `defective` run: all 20 defects were reported, but HYG-003 (a tracked `.pyc`) was
recorded with `file: null`. At that time `read_file` did not register a binary file as inspected, the evidence check rejected
every attempt to cite the `.pyc`, and the model spent about half of its turns on it before falling back to a location-less
finding. The scorer therefore counted one FP and one FN for what is one genuine defect. `read_file` and
`inspect_tracked_artifacts` now register such files, with regression tests; **that change has not been re-run against the
live model.**

## 10. Known limitations

* The current integration requires `OPENROUTER_API_KEY`. The assignment lets the evaluator configure supplied credentials
  but does not name their provider; a non-OpenRouter key alone cannot run this version.
* Gemini and the OpenRouter free router completed live audits before this refactor. The OpenRouter healthy run used six
  routed models, so it does not measure full-audit behaviour of any one pinned free model. Free models can be throttled or
  withdrawn at any time. The refactor itself was checked with mocked responses, not a new live audit.
* Detection quality has been measured on the author's own fixtures: Gemini on the defective fixture and the OpenRouter
  free router on the healthy fixture. Behaviour and false-positive rates on unseen repositories remain unmeasured.
* OpenRouter's free tier has per-minute and per-day request limits. The agent stops cleanly when an identified daily limit
  is hit (exit 4, labeled partial output) but cannot work around it.
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
agent.py  tools.py       assignment-facing CLI and inspection-tool entry points
auditor/                 internal implementation package
tests/                  pytest suite (scripted fake model, fixture repositories, no network)
eval/                   fixture generator with answer keys, and the scorer
scripts/                helper script to regenerate the sample report
sample_report.json      authentic example output (completed audit of octocat/Hello-World)
docs/sample_report.trace.json  trace of that run
.env.example            variable names only, no credentials
requirements.txt  pyproject.toml  .github/workflows/ci.yml
```
