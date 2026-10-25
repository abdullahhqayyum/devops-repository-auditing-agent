from fakes import ScriptedLLM, call

from auditor import prompts
from auditor.agent_loop import AuditAgent, AuditConfig
from auditor.llm_client import ToolCall
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


def test_context_trimming_preserves_tool_output_for_recorded_finding(sample_repo):
    agent = AuditAgent(
        ScriptedLLM([]),
        RepoTools(sample_repo),
        "u",
        AuditConfig(context_char_budget=500, keep_recent_tool_results=0, self_review=False),
    )
    agent.tools.read_file("app/config.py")
    agent.store.add(SECRET_FINDING)
    agent.messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "finding-read", "name": "read_file", "arguments": {"path": "app/config.py"}}],
        },
        {"role": "tool", "tool_call_id": "finding-read", "name": "read_file", "content": "evidence " * 500},
        {"role": "tool", "tool_call_id": "other", "name": "read_file", "content": "exploration " * 500},
    ]

    assert agent._trim_context()
    assert not agent.messages[1]["content"].startswith("[elided")
    assert agent.messages[2]["content"].startswith("[elided")


def test_compacted_evidence_can_be_reread_before_recording_finding(sample_repo, monkeypatch):
    """A later finding remains grounded when its first inspection was compacted."""
    import auditor.agent_loop as agent_loop

    capped_output = "\n".join(f"{line}: insecure setting" for line in range(1, 1201))[:12_000]
    calls = []

    def capped_dispatch(tools, name, args):
        calls.append((name, args))
        if name == "read_file":
            tools.observed_files.add(args["path"])
            return capped_output
        return ""

    monkeypatch.setattr(agent_loop, "dispatch", capped_dispatch)

    def decide_after_compaction(messages):
        config_result = next(
            (message["content"] for message in messages if message.get("tool_call_id") == "config-read"),
            "",
        )
        reread_result = next(
            (message["content"] for message in messages if message.get("tool_call_id") == "config-reread"),
            "",
        )
        if reread_result == capped_output:
            return [
                call(
                    "record_finding",
                    rule_id="SEC-001",
                    category="security",
                    severity="high",
                    file="app/config.py",
                    line=3,
                    description="Insecure setting confirmed after rereading the file",
                    evidence="3: insecure setting",
                    recommendation="Replace the insecure setting with a secure configuration",
                )
            ]
        if config_result.startswith("[elided"):
            return [ToolCall("config-reread", "read_file", {"path": "app/config.py"})]
        return [call("read_file", path="app/config.py")]

    llm = ScriptedLLM(
        [
            [ToolCall("config-read", "read_file", {"path": "app/config.py"})],
            *[[call("read_file", path="app/main.py")] for _ in range(8)],
            decide_after_compaction,
            decide_after_compaction,
            mark_all(),
            [call("finish_audit")],
        ]
    )
    agent = AuditAgent(
        llm,
        RepoTools(sample_repo),
        "u",
        AuditConfig(
            context_char_budget=20_000,
            keep_recent_tool_results=1,
            cache_inspection_results=False,
            self_review=False,
        ),
    )

    findings = agent.run()

    assert findings and findings[0]["file"] == "app/config.py"
    assert findings[0]["line"] == 3
    assert any(args == {"path": "app/config.py"} for _, args in calls[1:])
    assert any(e["type"] == "control" and e["message"] == "context trimmed" for e in agent.trace.events)
    assistant_ids = {
        call["id"] for message in agent.messages if message.get("role") == "assistant" for call in message.get("tool_calls", [])
    }
    tool_ids = {message["tool_call_id"] for message in agent.messages if message.get("role") == "tool"}
    assert assistant_ids == tool_ids
    assert set(prompts.CATEGORIES) == set(agent.reviewed)


def test_default_context_policy_keeps_four_recent_tool_results():
    assert AuditConfig().context_char_budget == 100_000
    assert AuditConfig().keep_recent_tool_results == 4


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


def test_system_prompt_guides_batching_early_recording_and_negative_observations():
    text = prompts.SYSTEM_PROMPT
    assert "SAME turn" in text and "instead of one per" in text  # batch independent calls
    assert "Go one step at a time only when the next call depends" in text  # but stay sequential when it does
    assert "Record as you go" in text and "instead of holding everything back" in text
    assert "Skip tools that cannot apply" in text
    assert "Absence matters too" in text and "acceptable equivalent" in text
    assert "together in one turn" in prompts.initial_message("u", "o")  # the first message carries its own reminder


def test_prompt_guidance_is_generic_not_tied_to_any_repository():
    for text in (prompts.SYSTEM_PROMPT, prompts.initial_message("u", "o")):
        for demo_specific in ("jsmith", "Inventory", "tinycalc", "greet.yml", "prod-db.internal"):
            assert demo_specific not in text


def test_batched_calls_in_one_turn_are_all_executed_and_answered_in_order(sample_repo):
    llm = ScriptedLLM(
        [
            [call("read_file", path="app/config.py"), call("read_file", path="app/main.py"), call("read_file", path="README.md")],
            mark_all(),
            [call("finish_audit")],
        ]
    )
    agent = AuditAgent(llm, RepoTools(sample_repo), "u", AuditConfig(self_review=False))
    agent.run()
    answers = [m for m in llm.seen_messages[1] if m["role"] == "tool"]
    assert len(answers) == 3
    assert [a["content"].split(" ", 1)[0] for a in answers] == ["app/config.py", "app/main.py", "README.md"]
    assert {"app/config.py", "app/main.py", "README.md"} <= agent.tools.observed_files
    assert agent.trace.events[-1]["steps"] == 3  # three files were read in a single model turn


def test_a_finding_can_be_recorded_in_the_same_turn_as_the_read_that_supports_it(sample_repo):
    llm = ScriptedLLM(
        [
            [
                call("read_file", path="app/config.py"),
                call("record_finding", **SECRET_FINDING),
                call("read_file", path="app/main.py"),
            ],
            mark_all(),
            [call("finish_audit")],
            [call("finish_audit")],
        ]
    )
    agent = AuditAgent(llm, RepoTools(sample_repo), "u")
    violations = agent.run()
    assert [v["rule_id"] for v in violations] == ["SEC-001"]
    findings = [e for e in agent.trace.events if e["type"] == "finding"]
    assert findings and findings[0]["result"].startswith("Recorded")  # accepted mid-investigation, not at the end
    assert [e["name"] for e in agent.trace.events if e["type"] == "tool_call"] == ["read_file", "read_file"]


def test_a_dependent_followup_still_works_after_a_batched_turn(sample_repo):
    def follow_up(messages):
        results = [m["content"] for m in messages if m["role"] == "tool"][-2:]
        assert any("verify=False" in r for r in results)  # the second turn reacts to what the batch returned
        return [call("search_code", pattern="verify=False")]

    llm = ScriptedLLM(
        [
            [call("read_file", path="app/main.py"), call("read_file", path="app/config.py")],
            follow_up,
            mark_all(),
            [call("finish_audit")],
        ]
    )
    agent = AuditAgent(llm, RepoTools(sample_repo), "u", AuditConfig(self_review=False))
    agent.run()
    names = [e["name"] for e in agent.trace.events if e["type"] == "tool_call"]
    assert names == ["read_file", "read_file", "search_code"]
