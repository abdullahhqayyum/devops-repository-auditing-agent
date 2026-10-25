"""Retry, quota and wall-clock handling for LLM requests. Everything is mocked: no network, no real sleeping."""

import json
import logging
import time as real_time
from types import SimpleNamespace

import pytest
from conftest import FAKE_KEY
from fakes import ScriptedLLM, call

import agent as cli
import llm_client
import prompts
from agent_loop import AuditAgent, AuditConfig
from llm_client import (
    RATE_LIMIT_ATTEMPTS,
    TRANSIENT_ATTEMPTS,
    LLMBudgetExhausted,
    LLMError,
    OpenRouterLLM,
    _classify,
    _retrying,
    _safe_message,
    _Throttle,
    _TransientError,
    build_llm,
)
from tools import RepoTools

DAY_ID = "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
MINUTE_ID = "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
GENERIC_QUOTA_TEXT = "You exceeded your current quota, please check your plan and billing details."


class Clock:
    """Stands in for the ``time`` module inside llm_client: sleeping just advances the clock."""

    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = Clock()
    monkeypatch.setattr(llm_client, "time", fake)
    monkeypatch.setattr(llm_client.random, "uniform", lambda _a, _b: 0.0)
    return fake


class Flaky:
    """A callable that raises the given errors in order, then returns ``result``."""

    def __init__(self, errors, result="ok"):
        self.errors = list(errors)
        self.result = result
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.result


def http_error(status, message="boom", body=None, headers=None):
    exc = Exception(message)
    exc.status_code = status
    exc.body = body
    exc.response = SimpleNamespace(headers=headers or {})
    return exc


def quota_body(*quota_ids, retry_delay=None, as_list=False):
    """A structured 429 body with QuotaFailure and optional RetryInfo, including upstream error shapes."""
    details = [
        {
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [
                {"quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests", "quotaId": q}
                for q in quota_ids
            ],
        }
    ]
    if retry_delay:
        details.append({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay})
    error = {"code": 429, "message": GENERIC_QUOTA_TEXT, "status": "RESOURCE_EXHAUSTED", "details": details}
    return [{"error": error}] if as_list else {"error": error}


# ------------------------------------------------------------ 1. single-layer retries
def test_sdk_client_does_not_retry_on_its_own(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    assert build_llm().client.max_retries == 0


def test_retry_policy_constants():
    assert RATE_LIMIT_ATTEMPTS == 3 and TRANSIENT_ATTEMPTS == 6


# ----------------------------------------------- 2/3. quota windows and body shapes
def test_per_day_quota_stops_after_one_request(clock):
    flaky = Flaky([http_error(429, GENERIC_QUOTA_TEXT, quota_body(DAY_ID))] * 5)
    with pytest.raises(LLMError, match="quota exhausted") as info:
        _retrying(flaky)
    assert flaky.calls == 1 and clock.sleeps == []
    assert DAY_ID in str(info.value) and "day-window quota" in str(info.value)


def test_per_minute_quota_is_retried_with_a_bounded_count(clock):
    flaky = Flaky([http_error(429, GENERIC_QUOTA_TEXT, quota_body(MINUTE_ID))] * 10)
    with pytest.raises(LLMError, match="after 3 attempt") as info:
        _retrying(flaky)
    assert flaky.calls == RATE_LIMIT_ATTEMPTS
    assert "quota exhausted" not in str(info.value) and "minute-window" in str(info.value)


def test_a_day_violation_wins_over_a_minute_violation(clock):
    flaky = Flaky([http_error(429, GENERIC_QUOTA_TEXT, quota_body(MINUTE_ID, DAY_ID))] * 5)
    with pytest.raises(LLMError, match="quota exhausted"):
        _retrying(flaky)
    assert flaky.calls == 1


def test_generic_billing_wording_alone_does_not_end_the_run(clock):
    """The same text is sent for per-minute limits, so without a structured window the 429 is retried (bounded)."""
    flaky = Flaky([http_error(429, GENERIC_QUOTA_TEXT, body=None)] * 10)
    with pytest.raises(LLMError, match="quota window not identified"):
        _retrying(flaky)
    assert flaky.calls == RATE_LIMIT_ATTEMPTS


def test_specific_per_day_wording_is_enough_to_stop(clock):
    message = "Quota exceeded for metric generate_content_free_tier_requests, limit: 20 requests per day"
    flaky = Flaky([http_error(429, message)] * 5)
    with pytest.raises(LLMError, match="quota exhausted"):
        _retrying(flaky)
    assert flaky.calls == 1


@pytest.mark.parametrize(
    ("body", "kind", "window"),
    [
        (quota_body(DAY_ID), "quota_exhausted", "day"),
        (quota_body(DAY_ID, as_list=True), "quota_exhausted", "day"),
        (quota_body(MINUTE_ID, as_list=True), "rate_limited", "minute"),
        (quota_body(MINUTE_ID), "rate_limited", "minute"),
        (None, "rate_limited", None),
        ("plain text body", "rate_limited", None),
        ([1, "x", None, [{"error": "not a dict"}]], "rate_limited", None),
        ({"error": "just a string"}, "rate_limited", None),
        ({"error": {"message": "Rate limit reached on tokens per min (TPM)"}}, "rate_limited", "minute"),
        ({"error": {"code": "rate_limit_exceeded"}}, "rate_limited", None),
    ],
)
def test_error_body_shapes_are_classified_without_crashing(body, kind, window):
    info = _classify(http_error(429, "rate limited", body))
    assert (info.kind, info.window) == (kind, window)


# ------------------------------------------------------------ 4. Structured insufficient_quota
@pytest.mark.parametrize(
    "body",
    [
        {"error": {"code": "insufficient_quota", "type": "insufficient_quota", "message": "You exceeded your current quota"}},
        [{"error": {"type": "insufficient_quota", "message": "billing"}}],
    ],
)
def test_structured_insufficient_quota_is_not_retried(clock, body):
    flaky = Flaky([http_error(429, "quota", body)] * 5)
    with pytest.raises(LLMError, match="insufficient_quota"):
        _retrying(flaky)
    assert flaky.calls == 1 and clock.sleeps == []


def test_plain_rate_limit_is_still_retried(clock):
    flaky = Flaky([http_error(429, "slow down", {"error": {"code": "rate_limit_exceeded"}})], result="ok")
    assert _retrying(flaky) == "ok" and flaky.calls == 2


# ------------------------------------------------------------------ 5. Retry-After
def test_retry_after_seconds_header_is_honoured(clock):
    assert _retrying(Flaky([http_error(429, headers={"retry-after": "7"})])) == "ok"
    assert clock.sleeps == [7.0]


def test_retry_after_ms_header_is_honoured(clock):
    _retrying(Flaky([http_error(429, headers={"retry-after-ms": "2500"})]))
    assert clock.sleeps == [2.5]


def test_retry_info_and_message_hints_are_used_when_no_header(clock):
    _retrying(Flaky([http_error(429, body=quota_body(MINUTE_ID, retry_delay="3s"))]))
    _retrying(Flaky([http_error(429, "Please retry in 4.5s.")]))
    assert clock.sleeps == [3.0, 4.5]


def test_invalid_retry_after_falls_back_to_exponential_backoff(clock):
    for bad in ("soon", "-5", "nan", ""):
        clock.sleeps.clear()
        _retrying(Flaky([http_error(429, headers={"retry-after": bad})]))
        assert clock.sleeps == [llm_client.RATE_LIMIT_BASE_DELAY], bad


def test_exponential_backoff_between_rate_limited_attempts(clock):
    with pytest.raises(LLMError):
        _retrying(Flaky([http_error(429)] * 10))
    assert clock.sleeps == [5.0, 10.0]


def test_retry_after_is_floored_and_an_excessive_one_ends_the_run(clock):
    _retrying(Flaky([http_error(429, headers={"retry-after": "0"})]))
    assert clock.sleeps == [1.0]
    clock.sleeps.clear()
    flaky = Flaky([http_error(429, headers={"retry-after": "3600"})] * 5)
    with pytest.raises(LLMError, match="longer than"):
        _retrying(flaky)
    assert flaky.calls == 1 and clock.sleeps == []


# ------------------------------------------- 6/7/8. attempt counts and transient recovery
def test_no_sleep_after_the_final_attempt(clock):
    flaky = Flaky([http_error(429)] * 10)
    with pytest.raises(LLMError):
        _retrying(flaky)
    assert flaky.calls == 3 and len(clock.sleeps) == 2
    clock.sleeps.clear()
    flaky = Flaky([http_error(503)] * 10)
    with pytest.raises(LLMError, match="after 6 attempt"):
        _retrying(flaky)
    assert flaky.calls == 6 and len(clock.sleeps) == 5
    assert clock.sleeps == [3.0, 6.0, 12.0, 24.0, 48.0]


def test_recovery_after_transient_503s(clock):
    flaky = Flaky([http_error(503), http_error(503)], result="recovered")
    assert _retrying(flaky) == "recovered"
    assert flaky.calls == 3 and clock.sleeps == [3.0, 6.0]


@pytest.mark.parametrize("error", [http_error(500), http_error(502), http_error(529), _TransientError("empty response")])
def test_eligible_temporary_failures_are_retried(clock, error):
    assert _retrying(Flaky([error])) == "ok"


def test_connection_errors_are_retried_by_name(clock):
    class APIConnectionError(Exception):
        pass

    assert _retrying(Flaky([APIConnectionError("reset")])) == "ok"


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_non_retryable_errors_fail_immediately(clock, status):
    flaky = Flaky([http_error(status, "nope")] * 3)
    with pytest.raises(LLMError, match="nope"):
        _retrying(flaky)
    assert flaky.calls == 1 and clock.sleeps == []


# --------------------------------------------------- 9. deadline during retries and requests
def test_retry_that_would_pass_the_deadline_stops_without_sleeping(clock):
    flaky = Flaky([http_error(429, headers={"retry-after": "10"})] * 5)
    with pytest.raises(LLMBudgetExhausted, match="wall-clock budget") as info:
        _retrying(flaky, deadline=clock.now + 4)
    assert flaky.calls == 1 and clock.sleeps == []
    assert isinstance(info.value, LLMError)


def test_retry_within_the_deadline_still_happens(clock):
    assert _retrying(Flaky([http_error(429, headers={"retry-after": "10"})]), deadline=clock.now + 100) == "ok"
    assert clock.sleeps == [10.0]


def test_throttle_does_not_wait_past_the_deadline(clock):
    throttle = _Throttle(10)
    throttle.wait()
    with pytest.raises(LLMBudgetExhausted):
        throttle.wait(deadline=clock.now + 5)
    assert clock.sleeps == []
    throttle.wait(deadline=clock.now + 500)
    assert clock.sleeps == [10.0]


class _CapturingCompletions:
    def __init__(self):
        self.kwargs = []

    def create(self, **kwargs):
        self.kwargs.append(kwargs)
        message = SimpleNamespace(content="ok", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=None)


def _openrouter_llm(capture, deadline=None):
    llm = OpenRouterLLM.__new__(OpenRouterLLM)
    llm.model, llm.max_tokens, llm.throttle, llm.timeout = "m", 100, _Throttle(0), 120.0
    llm.deadline = deadline
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=capture))
    return llm


def test_request_timeout_is_bounded_by_the_remaining_budget(clock):
    capture = _CapturingCompletions()
    _openrouter_llm(capture, deadline=clock.now + 30).chat("s", [{"role": "user", "content": "hi"}], [])
    assert capture.kwargs[0]["timeout"] == 30.0
    capture = _CapturingCompletions()
    _openrouter_llm(capture).chat("s", [{"role": "user", "content": "hi"}], [])
    assert capture.kwargs[0]["timeout"] == 120.0  # no deadline: the configured timeout


def test_no_request_is_sent_once_the_budget_is_gone(clock):
    capture = _CapturingCompletions()
    with pytest.raises(LLMBudgetExhausted):
        _openrouter_llm(capture, deadline=clock.now - 1).chat("s", [{"role": "user", "content": "hi"}], [])
    assert capture.kwargs == []


# --------------------------------------------------------------- 10. sensitive error text
def test_error_text_is_redacted_including_the_configured_api_key(monkeypatch, caplog):
    monkeypatch.setenv("GEMINI_API_KEY", "AQ.not-a-pattern-we-know-0123456789")  # a key format the pattern list misses
    nasty = f"bad request key=AQ.not-a-pattern-we-know-0123456789 aws {FAKE_KEY} password = 'Zq9xLm2pRt7vWn' db password: hunter2"
    text = _safe_message(Exception(nasty))
    for secret in ("AQ.not-a-pattern-we-know-0123456789", FAKE_KEY, "Zq9xLm2pRt7vWn", "hunter2"):
        assert secret not in text
    assert "[REDACTED:api_key]" in text

    clock_free_flaky = Flaky([http_error(503, nasty), http_error(401, nasty)])
    monkeypatch.setattr(llm_client.time, "sleep", lambda _s: None)
    with caplog.at_level(logging.INFO, logger="audit.llm"), pytest.raises(LLMError) as info:
        _retrying(clock_free_flaky)
    everything = caplog.text + str(info.value)
    for secret in ("AQ.not-a-pattern-we-know-0123456789", FAKE_KEY, "Zq9xLm2pRt7vWn", "hunter2"):
        assert secret not in everything


def test_retry_logs_status_attempt_delay_and_window_but_no_headers(clock, caplog):
    headers = {"retry-after": "7", "authorization": "Bearer placeholder-not-a-real-credential"}
    with caplog.at_level(logging.INFO, logger="audit.llm"):
        _retrying(Flaky([http_error(429, "slow", quota_body(MINUTE_ID), headers)]))
    line = caplog.text
    assert "HTTP 429" in line and "attempt 1/3" in line and "retrying in 7.0s" in line and "minute-window" in line
    assert "authorization" not in line.lower() and "placeholder-not-a-real-credential" not in line


# --------------------------------------------------------------- 11. agent-level behaviour
SECRET_FINDING = {
    "rule_id": "SEC-001",
    "category": "security",
    "severity": "critical",
    "file": "app/config.py",
    "line": 3,
    "description": "Hardcoded AWS access key",
    "evidence": "AWS key literal assigned to AWS_KEY",
    "recommendation": "Load from environment or a secrets manager and rotate the key",
}


class _FailingLLM:
    model = "failing-test-model"

    def __init__(self, error):
        self.error = error

    def chat(self, system, messages, tools):
        raise self.error


class _ThenExhausted(ScriptedLLM):
    """Plays its scripted turns, then reports that the wall-clock budget ran out (as the real adapters do)."""

    deadline = None

    def chat(self, system, messages, tools):
        if not self.turns:
            raise LLMBudgetExhausted("wall-clock budget exhausted: a 30s retry delay would pass the deadline")
        return super().chat(system, messages, tools)


def _run_cli(sample_repo, tmp_path, monkeypatch, llm):
    monkeypatch.setattr(cli, "build_llm", lambda *a, **k: llm)
    out = tmp_path / "report.json"
    code = cli.main(["--repo", str(sample_repo), "--allow-local", "--output", str(out)])
    return code, out, tmp_path / "report.json.partial", tmp_path / "report.trace.json"


def _mark_all():
    return [call("mark_reviewed", category=c, status="no_issues", note="ok") for c in prompts.CATEGORIES]


def test_agent_propagates_llm_errors_and_records_why(sample_repo):
    agent = AuditAgent(_FailingLLM(LLMError("quota exhausted")), RepoTools(sample_repo), "u")
    with pytest.raises(LLMError, match="quota exhausted"):
        agent.run()
    assert agent.stop_reason == "llm_error"
    assert agent.trace.events[-1]["type"] == "finished" and agent.trace.events[-1]["stop_reason"] == "llm_error"


def test_timeout_before_any_inspection_is_incomplete_with_exit_4(sample_repo, tmp_path, monkeypatch, capsys):
    code, report, partial, trace_path = _run_cli(sample_repo, tmp_path, monkeypatch, _ThenExhausted([]))
    assert code == cli.EXIT_LLM != cli.EXIT_OK
    assert not report.exists()  # never presented as a finished audit
    saved = json.loads(partial.read_text())
    assert saved["incomplete"] is True and saved["violations"] == []
    assert saved["stop_reason"].startswith("time budget exhausted")
    err = capsys.readouterr().err
    assert "time budget exhausted" in err and "investigation incomplete" in err and "report.json.partial" in err
    trace = json.loads(trace_path.read_text())
    assert trace["status"] == "incomplete" and trace["stop_reason"] == "wall_clock_budget_exhausted"
    assert trace["tool_call_counts"] == {}  # nothing was inspected, and the trace says so


def test_timeout_after_findings_keeps_them_in_the_labeled_partial_file(sample_repo, tmp_path, monkeypatch):
    llm = _ThenExhausted([[call("read_file", path="app/config.py")], [call("record_finding", **SECRET_FINDING)]])
    code, report, partial, trace_path = _run_cli(sample_repo, tmp_path, monkeypatch, llm)
    assert code == cli.EXIT_LLM and not report.exists()
    saved = json.loads(partial.read_text())
    assert saved["incomplete"] is True
    assert [v["rule_id"] for v in saved["violations"]] == ["SEC-001"]
    trace = json.loads(trace_path.read_text())
    assert trace["status"] == "incomplete"
    finished = trace["events"][-1]
    assert (
        finished["type"] == "finished" and finished["findings"] == 1 and finished["stop_reason"] == "wall_clock_budget_exhausted"
    )
    control = [e for e in trace["events"] if e["type"] == "control" and "incomplete" in e["message"]]
    assert control and control[0]["findings_kept"] == 1 and control[0]["wall_clock_budget_seconds"] == 1500
    assert "wall-clock budget exhausted" in control[0]["detail"]


class _FakeWallClock:
    """Replaces ``time`` inside agent_loop; anchored at the real time because Trace.started uses the real clock."""

    def __init__(self):
        self.now = real_time.time()

    def time(self):
        return self.now

    def monotonic(self):
        return real_time.monotonic()


def test_wall_clock_expiring_between_turns_stops_without_a_wrapup_turn(sample_repo, monkeypatch):
    clock = _FakeWallClock()
    monkeypatch.setattr("agent_loop.time", clock)

    class SlowFirstTurn(ScriptedLLM):
        def chat(self, system, messages, tools):
            response = super().chat(system, messages, tools)
            clock.now += 100  # this model turn "took" 100 s
            return response

    llm = SlowFirstTurn([[call("read_file", path="app/config.py")], [call("record_finding", **SECRET_FINDING)], _mark_all()])
    agent = AuditAgent(llm, RepoTools(sample_repo), "u", AuditConfig(wall_clock_seconds=50, self_review=False))
    with pytest.raises(LLMBudgetExhausted, match="50s wall-clock budget"):
        agent.run()
    assert len(llm.seen_messages) == 1  # no further model turn, in particular no extra wrap-up turn
    assert agent.stop_reason == "wall_clock_budget_exhausted"
    assert not agent.in_wrapup


def test_step_budget_still_gets_a_normal_wrapup_and_completes(sample_repo, tmp_path, monkeypatch):
    llm = ScriptedLLM([[call("list_files")], [call("list_files")], [call("finish_audit")]])
    monkeypatch.setattr(cli, "build_llm", lambda *a, **k: llm)
    out = tmp_path / "report.json"
    assert cli.main(["--repo", str(sample_repo), "--allow-local", "--output", str(out), "--max-steps", "1"]) == cli.EXIT_OK
    assert json.loads(out.read_text()) == {"repository": str(sample_repo.resolve()), "violations": []}


def test_normal_completion_before_the_deadline_exits_zero_with_a_valid_report(sample_repo, tmp_path, monkeypatch):
    llm = _ThenExhausted(
        [
            [call("read_file", path="app/config.py")],
            [call("record_finding", **SECRET_FINDING)],
            _mark_all(),
            [call("finish_audit")],
            [call("finish_audit")],
        ]
    )
    code, report, partial, trace_path = _run_cli(sample_repo, tmp_path, monkeypatch, llm)
    assert code == cli.EXIT_OK and not partial.exists()
    data = json.loads(report.read_text())
    assert set(data) == {"repository", "violations"} and [v["rule_id"] for v in data["violations"]] == ["SEC-001"]
    trace = json.loads(trace_path.read_text())
    assert trace["status"] == "complete" and "stop_reason" not in trace
    assert trace["events"][-1]["stop_reason"] is None
    assert llm.deadline is not None  # the agent did give the adapter a deadline


def test_a_recovered_temporary_rate_limit_does_not_fail_the_audit(sample_repo, tmp_path, monkeypatch, clock):
    class RateLimitedOnce(ScriptedLLM):
        def __init__(self, turns):
            super().__init__(turns)
            self.failures = [http_error(429, "slow down", quota_body(MINUTE_ID), {"retry-after": "2"})]

        def chat(self, system, messages, tools):
            def attempt():
                if self.failures:
                    raise self.failures.pop(0)
                return ScriptedLLM.chat(self, system, messages, tools)

            return _retrying(attempt)

    llm = RateLimitedOnce(
        [
            [call("read_file", path="app/config.py")],
            [call("record_finding", **SECRET_FINDING)],
            _mark_all(),
            [call("finish_audit")],
            [call("finish_audit")],
        ]
    )
    code, report, partial, trace_path = _run_cli(sample_repo, tmp_path, monkeypatch, llm)
    assert code == cli.EXIT_OK and report.exists() and not partial.exists()
    assert clock.sleeps == [2.0]  # waited for the server's hint exactly once
    assert json.loads(trace_path.read_text())["status"] == "complete"


def test_cli_reports_a_plain_llm_failure_with_exit_4_and_a_labeled_partial_file(sample_repo, tmp_path, monkeypatch, capsys):
    llm = _FailingLLM(LLMError("LLM quota exhausted (HTTP 429)"))
    code, report, partial, trace_path = _run_cli(sample_repo, tmp_path, monkeypatch, llm)
    assert code == cli.EXIT_LLM and not report.exists()
    saved = json.loads(partial.read_text())
    assert saved["incomplete"] is True and saved["stop_reason"].startswith("LLM failure")
    assert "quota exhausted" in capsys.readouterr().err
    trace = json.loads(trace_path.read_text())
    assert trace["status"] == "incomplete" and trace["stop_reason"] == "llm_error"


def test_trace_and_partial_files_never_contain_secrets_from_the_error(sample_repo, tmp_path, monkeypatch):
    nasty = f"timeout while sending key {FAKE_KEY} password = 'Zq9xLm2pRt7vWn'"
    code, _report, partial, trace_path = _run_cli(sample_repo, tmp_path, monkeypatch, _FailingLLM(LLMBudgetExhausted(nasty)))
    assert code == cli.EXIT_LLM
    for text in (partial.read_text(), trace_path.read_text()):
        assert FAKE_KEY not in text and "Zq9xLm2pRt7vWn" not in text


def test_agent_gives_the_llm_a_deadline_and_never_extends_it(sample_repo):
    class Recording(ScriptedLLM):
        deadline = None

        def chat(self, system, messages, tools):
            self.seen_deadlines = [*getattr(self, "seen_deadlines", []), self.deadline]
            return super().chat(system, messages, tools)

    llm = Recording([[call("list_files")], [call("list_files")], [call("finish_audit")]])
    config = AuditConfig(max_steps=1, wrapup_steps=2, wall_clock_seconds=300, self_review=False)
    AuditAgent(llm, RepoTools(sample_repo), "u", config).run()
    first, second = llm.seen_deadlines[:2]
    assert first is not None and first <= real_time.monotonic() + 300
    assert second == first  # wrap-up runs inside the configured budget: no undocumented grace period


def test_llms_without_a_deadline_attribute_are_left_alone(sample_repo):
    reviewed = [call("mark_reviewed", category=c, status="no_issues", note="x") for c in prompts.CATEGORIES]
    llm = ScriptedLLM([reviewed, [call("finish_audit")]])
    agent = AuditAgent(llm, RepoTools(sample_repo), "u", AuditConfig(self_review=False))
    agent.run()
    assert agent.finished and not hasattr(llm, "deadline")
