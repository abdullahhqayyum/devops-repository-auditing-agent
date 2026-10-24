#!/usr/bin/env python3
"""DevOps repository auditing agent.

Usage:
    python agent.py --repo https://github.com/<owner>/<repo> --output report.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from agent_loop import AuditAgent, AuditConfig
from llm_client import LLMBudgetExhausted, LLMError, build_llm
from redaction import redact
from repo_loader import RepoError, cleanup, clone_repository, parse_github_url, prepare_local_repository
from report import build_report, validate_report, write_json_atomic
from tools import RepoTools

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_REPO = 3
EXIT_LLM = 4
EXIT_INTERNAL = 5


class RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI agent that audits a public GitHub repository for DevOps violations.")
    p.add_argument("--repo", required=True, help="Public GitHub repository URL")
    p.add_argument("--output", required=True, help="Path of the JSON report to write")
    p.add_argument("--trace", default=None, help="Tool-call trace path (default: <output>.trace.json; 'none' to disable)")
    p.add_argument(
        "--provider",
        default=None,
        choices=["anthropic", "openai", "gemini", "ollama"],
        help="LLM provider (default: env LLM_PROVIDER or auto)",
    )
    p.add_argument("--model", default=None, help="Model name (default: env LLM_MODEL or provider default)")
    p.add_argument("--max-steps", type=int, default=45, help="Maximum LLM investigation turns")
    p.add_argument("--max-tool-calls", type=int, default=120, help="Maximum inspection tool executions")
    p.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Wall-clock budget in seconds (default 1500; 3600 for local Ollama). Running out ends the audit as incomplete "
        "(exit 4, findings so far in <output>.partial). A request already in flight can overrun it slightly.",
    )
    p.add_argument(
        "--sandbox",
        choices=["none", "docker"],
        default="none",
        help="Enable run_tests inside a Docker sandbox (default: none; repository code is never executed)",
    )
    p.add_argument("--allow-local", action="store_true", help="Allow --repo to be a local git path (development/eval only)")
    p.add_argument("--keep-clone", action="store_true", help="Do not delete the temporary checkout (debugging)")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging to stderr")
    return p.parse_args(argv)


def setup_logging(verbose: bool) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(RedactingFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.INFO if verbose else logging.WARNING)
    logging.getLogger("audit").setLevel(logging.INFO)
    for noisy in ("httpx", "httpcore", "anthropic", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def fail(code: int, message: str) -> int:
    print(f"error: {redact(message)}", file=sys.stderr)
    return code


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    log = logging.getLogger("audit.cli")

    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
    except ImportError:
        pass

    # 1. Validate input before spending anything.
    local = args.allow_local and Path(args.repo).expanduser().exists()
    if local:
        repo_label = str(Path(args.repo).expanduser().resolve())
    else:
        try:
            repo_label, _ = parse_github_url(args.repo)
        except RepoError as exc:
            return fail(EXIT_USAGE, str(exc))
    output = Path(args.output)
    if output.exists() and output.is_dir():
        return fail(EXIT_USAGE, f"--output {output} is a directory")
    if args.max_steps < 1 or args.max_tool_calls < 1:
        return fail(EXIT_USAGE, "--max-steps and --max-tool-calls must be positive")

    # 2. LLM client (fails fast on missing credentials).
    try:
        llm = build_llm(args.provider, args.model)
    except LLMError as exc:
        return fail(EXIT_USAGE, str(exc))
    except ImportError as exc:
        return fail(EXIT_USAGE, f"LLM SDK not installed: {exc}. Run: pip install -r requirements.txt")

    # 3. Isolated read-only checkout.
    checkout = None
    try:
        log.info("Cloning %s", repo_label)
        checkout = prepare_local_repository(args.repo) if local else clone_repository(args.repo)
        context_chars = getattr(llm, "context_chars", 220_000)
        tools = RepoTools(
            checkout,
            sandbox=None if args.sandbox == "none" else args.sandbox,
            max_output_chars=12_000 if context_chars >= 150_000 else 6_000,
        )
        config = AuditConfig(
            max_steps=args.max_steps,
            max_tool_calls=args.max_tool_calls,
            wall_clock_seconds=args.timeout or (3600 if type(llm).__name__ == "OllamaLLM" else 1500),
            context_char_budget=context_chars,
            keep_recent_tool_results=12 if context_chars >= 150_000 else 5,
        )
        agent = AuditAgent(llm, tools, repo_label, config)
        log.info("Auditing with model %s", llm.model)

        # 4. Agent loop.
        try:
            violations = agent.run()
        except LLMError as exc:
            # Incomplete audit: keep what was found in a clearly labeled partial file; the report path stays untouched.
            reason = "time budget exhausted" if isinstance(exc, LLMBudgetExhausted) else "LLM failure"
            partial = {
                **build_report(repo_label, agent.store.as_report_list()),
                "incomplete": True,
                "stop_reason": redact(f"{reason}: {exc}")[:500],
            }
            write_json_atomic(output.with_suffix(output.suffix + ".partial"), partial)
            _write_trace(args, output, agent)
            return fail(EXIT_LLM, f"{reason}, investigation incomplete ({exc}). Partial findings saved to {output}.partial")

        # 5. Contract-checked output.
        report = build_report(repo_label, violations)
        validate_report(report)
        write_json_atomic(output, report)
        _write_trace(args, output, agent)
        print(
            f"Wrote {output} with {len(violations)} violation(s) "
            f"[{agent.inspection_calls} tool calls, {len(agent.trace.events)} trace events]",
            file=sys.stderr,
        )
        return EXIT_OK
    except RepoError as exc:
        return fail(EXIT_REPO, str(exc))
    except KeyboardInterrupt:
        return fail(130, "interrupted")
    except Exception as exc:  # noqa: BLE001 - last-resort guard so the CLI never dumps secrets in a traceback
        log.debug("internal error", exc_info=True)
        return fail(EXIT_INTERNAL, f"internal error: {type(exc).__name__}: {exc}")
    finally:
        if checkout is not None and not args.keep_clone:
            cleanup(checkout)


def _write_trace(args: argparse.Namespace, output: Path, agent: AuditAgent) -> None:
    if args.trace and args.trace.lower() == "none":
        return
    path = Path(args.trace) if args.trace else output.with_name(output.stem + ".trace.json")
    try:
        status = {"status": "incomplete", "stop_reason": agent.stop_reason} if agent.stop_reason else {"status": "complete"}
        write_json_atomic(path, {"repository": agent.repo_url, "model": agent.llm.model, **status, **agent.trace.to_dict()})
    except OSError as exc:
        logging.getLogger("audit.cli").warning("could not write trace: %s", exc)


if __name__ == "__main__":
    sys.exit(main())
