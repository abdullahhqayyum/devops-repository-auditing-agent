#!/usr/bin/env bash
# Regenerate sample_report.json (and its trace) with a real LLM run.
# Usage: scripts/generate_sample_report.sh [github_repo_url]
# With no URL, the bundled demo repository with injected defects is built and audited locally.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -ge 1 ]]; then
  python agent.py --repo "$1" --output sample_report.json --trace sample_report.trace.json -v
else
  python eval/make_demo_repo.py --out /tmp/demo-repos
  python agent.py --allow-local --repo /tmp/demo-repos/defective \
    --output sample_report.json --trace sample_report.trace.json -v
  python eval/score.py sample_report.json /tmp/demo-repos/defective.key.json
fi
