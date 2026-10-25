import json

from auditor import prompts
from auditor.agent_loop import AuditAgent, AuditConfig
from auditor.llm_client import LLMResponse
from tools import RepoTools


class TelemetryLLM:
    model = "telemetry-test-model"
    context_chars = 100_000

    def __init__(self, responses):
        self.responses = list(responses)

    def chat(self, system, messages, tools):
        return self.responses.pop(0)


def _mark_all():
    from fakes import call

    return [call("mark_reviewed", category=category, status="no_issues", note="checked") for category in prompts.CATEGORIES]


def test_trace_records_safe_per_turn_telemetry(sample_repo):
    from fakes import call

    llm = TelemetryLLM(
        [
            LLMResponse(
                "inspect",
                [call("read_file", path="app/config.py"), call("list_files")],
                input_tokens=120,
                output_tokens=15,
                request_duration_seconds=1.25,
                retry_count=2,
                throttle_wait_seconds=3.5,
                model="provider-model",
            ),
            LLMResponse("review", _mark_all(), input_tokens=80, output_tokens=10),
            LLMResponse("finish", [call("finish_audit")], input_tokens=40, output_tokens=5),
        ]
    )
    agent = AuditAgent(llm, RepoTools(sample_repo), "u", AuditConfig(self_review=False))
    agent.run()

    telemetry = [event for event in agent.trace.events if event["type"] == "llm_telemetry"]
    assert telemetry[0]["step"] == 1
    assert telemetry[0]["model"] == "provider-model"
    assert telemetry[0]["system_prompt_chars"] == len(prompts.SYSTEM_PROMPT)
    assert telemetry[0]["tool_schema_chars"] > 0
    assert telemetry[0]["conversation_history_chars"] > 0
    assert telemetry[0]["approximate_request_chars"] == (
        telemetry[0]["system_prompt_chars"] + telemetry[0]["tool_schema_chars"] + telemetry[0]["conversation_history_chars"]
    )
    assert telemetry[0]["input_tokens"] == 120
    assert telemetry[0]["output_tokens"] == 15
    assert telemetry[0]["tool_call_count"] == 2
    assert telemetry[0]["tool_result_chars"] > 0
    assert telemetry[0]["request_duration_seconds"] == 1.25
    assert telemetry[0]["retry_count"] == 2
    assert telemetry[0]["throttle_wait_seconds"] == 3.5
    assert telemetry[0]["context_compacted"] is False


def test_missing_provider_metadata_is_null_and_trace_contains_no_raw_content(sample_repo):
    from fakes import call

    secret = "sk-or-v1-" + "not-a-real-secret"
    llm = TelemetryLLM(
        [
            LLMResponse("", [call("search_code", pattern=secret)]),
            LLMResponse("", _mark_all()),
            LLMResponse("", [call("finish_audit")]),
        ]
    )
    agent = AuditAgent(llm, RepoTools(sample_repo), "u", AuditConfig(self_review=False))
    agent.run()
    serialized = json.dumps(agent.trace.to_dict())
    telemetry = [event for event in agent.trace.events if event["type"] == "llm_telemetry"]
    assert telemetry[0]["retry_count"] is None
    assert telemetry[0]["throttle_wait_seconds"] is None
    assert telemetry[0]["model"] == "telemetry-test-model"
    assert secret not in serialized
    assert prompts.SYSTEM_PROMPT not in serialized
    assert "conversation_history_chars" in serialized


def test_duplicate_read_only_inspection_reuses_prior_result(sample_repo):
    from fakes import ScriptedLLM, call

    llm = ScriptedLLM(
        [
            [call("list_files")],
            [call("list_files")],
            _mark_all(),
            [call("finish_audit")],
        ]
    )
    agent = AuditAgent(llm, RepoTools(sample_repo), "u", AuditConfig(self_review=False))
    agent.run()

    calls = [event for event in agent.trace.events if event["type"] == "tool_call" and event["name"] == "list_files"]
    assert len(calls) == 2
    assert calls[0]["cache_hit"] is False
    assert calls[1]["cache_hit"] is True
    assert "Cached result" in llm.seen_messages[2][-1]["content"]
