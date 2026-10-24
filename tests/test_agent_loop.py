from fakes import ScriptedLLM, call

import prompts
from agent_loop import AuditAgent, AuditConfig
from tools import RepoTools

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


def mark_all():
    return [call("mark_reviewed", category=c, status="no_issues", note="checked") for c in prompts.CATEGORIES]


def last_tool_results(messages):
    out = []
    for m in reversed(messages):
        if m["role"] != "tool":
            break
        out.append(m["content"])
    return out


def test_loop_executes_tools_and_records_finding(sample_repo):
    llm = ScriptedLLM(
        [
            [call("list_files")],
            [call("read_file", path="app/config.py")],
            [call("record_finding", **SECRET_FINDING)],
            mark_all(),
            [call("finish_audit", summary="done")],
            [call("finish_audit", summary="done")],  # after self-review nudge
        ]
    )
    agent = AuditAgent(llm, RepoTools(sample_repo), "https://github.com/x/y")
    violations = agent.run()
    assert [v["rule_id"] for v in violations] == ["SEC-001"]
    assert agent.finished
    names = [e["name"] for e in agent.trace.events if e["type"] == "tool_call"]
    assert names == ["list_files", "read_file"]
    # Tool output was fed back to the model.
    assert any("app/config.py" in m.get("content", "") for m in llm.seen_messages[1] if m["role"] == "tool")


def test_dynamic_follow_up_based_on_tool_output(sample_repo):
    """The next action depends on what the previous tool returned."""

    def react(messages):
        result = last_tool_results(messages)[0]
        line = next(ln for ln in result.splitlines() if "verify=False" in ln)
        path = line.split(":", 1)[0]
        return [call("read_file", path=path)]

    llm = ScriptedLLM([[call("search_code", pattern="verify=False")], react, mark_all(), [call("finish_audit")]])
    agent = AuditAgent(llm, RepoTools(sample_repo), "u", AuditConfig(self_review=False))
    agent.run()
    reads = [e["arguments"] for e in agent.trace.events if e["type"] == "tool_call" and e["name"] == "read_file"]
    assert reads == [{"path": "app/main.py"}]


def test_finding_without_inspection_is_rejected(sample_repo):
    llm = ScriptedLLM([[call("record_finding", **SECRET_FINDING)], mark_all(), [call("finish_audit")]])
    agent = AuditAgent(llm, RepoTools(sample_repo), "u", AuditConfig(self_review=False))
    assert agent.run() == []
    rejected = [e for e in agent.trace.events if e["type"] == "finding"]
    assert "REJECTED" in rejected[0]["result"]


def test_coverage_nudge_before_finishing(sample_repo):
    llm = ScriptedLLM([[call("finish_audit")], [call("finish_audit")]])
    agent = AuditAgent(llm, RepoTools(sample_repo), "u", AuditConfig(self_review=False))
    agent.run()
    second_turn_tool_msgs = [m for m in llm.seen_messages[1] if m["role"] == "tool"]
    assert "not been marked reviewed" in second_turn_tool_msgs[-1]["content"]
    assert agent.finished


def test_budget_exhaustion_enters_wrapup_and_blocks_inspection(sample_repo):
    llm = ScriptedLLM([[call("list_files")]] * 3 + [[call("list_files")], [call("finish_audit")]])
    agent = AuditAgent(llm, RepoTools(sample_repo), "u", AuditConfig(max_steps=2, wrapup_steps=3))
    agent.run()
    assert agent.inspection_calls == 2
    assert agent.finished
    assert any("budget" in str(e.get("message", "")) for e in agent.trace.events if e["type"] == "control")


def test_unknown_tool_and_bad_args_are_reported_not_fatal(sample_repo):
    llm = ScriptedLLM([[call("delete_repo"), call("read_file")], mark_all(), [call("finish_audit")]])
    agent = AuditAgent(llm, RepoTools(sample_repo), "u", AuditConfig(self_review=False))
    agent.run()
    tool_msgs = [m["content"] for m in llm.seen_messages[1] if m["role"] == "tool"]
    assert any("unknown tool" in t for t in tool_msgs)
    assert any("missing required" in t for t in tool_msgs)


def test_model_that_stops_calling_tools_terminates(sample_repo):
    agent = AuditAgent(ScriptedLLM([]), RepoTools(sample_repo), "u")
    assert agent.run() == []


def test_context_trimming_keeps_pairs(sample_repo):
    llm = ScriptedLLM([[call("read_file", path="app/config.py")] for _ in range(6)] + [mark_all(), [call("finish_audit")]])
    agent = AuditAgent(
        llm, RepoTools(sample_repo), "u", AuditConfig(context_char_budget=500, keep_recent_tool_results=1, self_review=False)
    )
    agent.run()
    tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
    calls = [c for m in agent.messages if m["role"] == "assistant" for c in m["tool_calls"]]
    assert len(tool_msgs) == len(calls)


def test_trace_never_contains_secrets_from_tool_arguments_or_reasons(sample_repo):
    import json

    from conftest import FAKE_KEY

    pw = "Zq9xLm2pRt7vWn"  # synthetic
    llm = ScriptedLLM(
        [
            [call("search_code", pattern=FAKE_KEY)],
            [call("read_file", path="app/config.py")],
            [call("record_finding", **SECRET_FINDING)],
            [call("retract_finding", finding_id="F1", reason=f"saw {FAKE_KEY} and password = '{pw}' in the log")],
            [call("no_such_tool", data={"nested": [f"password = '{pw}'", FAKE_KEY]})],
            mark_all(),
            [call("finish_audit")],
        ]
    )
    agent = AuditAgent(llm, RepoTools(sample_repo), "u", AuditConfig(self_review=False))
    agent.run()
    trace = json.dumps(agent.trace.to_dict())
    assert FAKE_KEY not in trace and pw not in trace
    # Still useful for debugging: the call, its tool name and the redaction marker are visible.
    searches = [e for e in agent.trace.events if e["type"] == "tool_call" and e["name"] == "search_code"]
    assert searches[0]["arguments"] == {"pattern": "[REDACTED:aws_access_key_id]"}
    assert "F1" in agent.trace.events[-1]["retracted"]


def test_malformed_tool_arguments_do_not_end_the_run_and_the_model_sees_the_error(sample_repo):
    def retry_after_error(messages):
        error = last_tool_results(messages)[0]
        assert error.startswith("ERROR: invalid argument(s) for list_files: 'path' expected string, got object")
        return [call("list_files", path="app")]  # the model corrects itself and carries on

    llm = ScriptedLLM([[call("list_files", path={"dir": "app"})], retry_after_error, mark_all(), [call("finish_audit")]])
    agent = AuditAgent(llm, RepoTools(sample_repo), "u", AuditConfig(self_review=False))
    agent.run()
    assert agent.finished
    calls = [e for e in agent.trace.events if e["type"] == "tool_call"]
    assert [c["name"] for c in calls] == ["list_files", "list_files"]
    assert "app/config.py" in calls[1]["output_preview"]
