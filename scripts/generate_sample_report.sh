#!/usr/bin/env bash
# Regenerate docs/sample_report.json (and its trace) with a real LLM run against a PUBLIC GitHub repository.
# Usage: scripts/generate_sample_report.sh [github_repo_url]   (default: https://github.com/octocat/Hello-World)
# Needs an OpenRouter key (see .env.example). If the run is incomplete (nonzero exit) no docs/sample_report.json is
# written, so a partial run can never replace the sample. The local evaluation fixtures are deliberately not used here:
# their "repository" value would be a filesystem path, not a public GitHub URL.
set -euo pipefail
cd "$(dirname "$0")/.."

repo="${1:-https://github.com/octocat/Hello-World}"
python agent.py --repo "$repo" --output docs/sample_report.json --trace docs/sample_report.trace.json -v
