"""Build local practice repositories with known, injected DevOps defects.

    python eval/make_demo_repo.py --out /tmp/demo-repos

Creates two git repositories plus answer keys:
  defective/   obvious and subtle defects (answer key: defective.key.json)
  healthy/     a mostly healthy project used to measure false positives (healthy.key.json)

Audit them with:  python agent.py --allow-local --repo /tmp/demo-repos/defective --output out.json
Score with:       python eval/score.py out.json /tmp/demo-repos/defective.key.json
The fake credentials below are syntactically plausible but not real.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

FAKE_AWS = "AKIA" + "J7Q2X9LMZ4TRB8KD"  # split so this file itself does not look like a leak
FAKE_GH = "ghp_" + "x8Kq2LmN4pR7sT1vW3yZ5aB6cD9eF0gH2jK4"

DEFECTIVE = {
    "README.md": "# Inventory Service\n\nRun `python app.py`.\n",
    ".gitignore": "*.log\n",
    "requirements.txt": "flask\nrequests>=2.0\nsqlalchemy==1.4.46\n",
    "setup.py": (
        "from setuptools import setup\n\n"
        "setup(name='inventory', version='0.1', install_requires=['flask==2.0.1', 'sqlalchemy==2.0.0'])\n"
    ),
    "app/__init__.py": "",
    "app/config.py": (
        "import os\n\n"
        "DEBUG = True\n"
        "DATABASE_URL = 'postgresql://admin:" + "S3cr3tPa55w0rd!" + "@prod-db.internal:5432/inventory'\n"
        f"AWS_ACCESS_KEY_ID = '{FAKE_AWS}'\n"
        "UPLOAD_DIR = '/Users/jsmith/projects/inventory/uploads'\n"
        "LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')\n"
    ),
    "app/main.py": (
        "import requests\n"
        "import yaml\n"
        "from flask import Flask\n\n"
        "from app.config import DEBUG\n\n"
        "app = Flask(__name__)\n\n\n"
        "@app.route('/health')\n"
        "def health():\n"
        "    try:\n"
        "        requests.get('https://status.example.com', verify=False, timeout=2)\n"
        "    except:\n"
        "        pass\n"
        "    return 'ok'\n\n\n"
        "if __name__ == '__main__':\n"
        "    app.run(debug=DEBUG)\n"
    ),
    "app/__pycache__/main.cpython-311.pyc": "\x00compiled\x00",
    "tests/test_main.py": "def test_health():\n    assert True\n",
    ".env": "SECRET_KEY=a8f5f167f44f4964e6c998dee827110c\n" + f"GITHUB_TOKEN={FAKE_GH}\n",
    ".vscode/settings.json": '{\n  "python.defaultInterpreterPath": "/Users/jsmith/.venvs/inv/bin/python"\n}\n',
    ".github/workflows/ci.yml": (
        "name: CI\n"
        "on:\n"
        "  push:\n"
        "    branches: [develop]\n"
        "permissions: write-all\n"
        "jobs:\n"
        "  test:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/checkout@v4\n"
        "      - uses: actions/setup-python@v5\n"
        "        with:\n"
        "          python-version: '3.11'\n"
        "      - run: pip install -r requirements.txt\n"
        "      - name: Lint\n"
        "        run: flake8 . || true\n"
        "      - name: Tests\n"
        "        run: pytest\n"
        "        continue-on-error: true\n"
    ),
    ".github/workflows/greet.yml": (
        "name: Greet\n"
        "on: pull_request_target\n"
        "jobs:\n"
        "  greet:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/checkout@v4\n"
        "        with:\n"
        "          ref: ${{ github.event.pull_request.head.sha }}\n"
        '      - run: echo "Thanks for ${{ github.event.pull_request.title }}"\n'
        "      - run: npm install && npm test\n"
        "        env:\n"
        "          NPM_TOKEN: ${{ secrets.NPM_TOKEN }}\n"
        "      - uses: some-org/labeler@master\n"
    ),
    "Dockerfile": 'FROM python:latest\nCOPY . /app\nRUN pip install -r /app/requirements.txt\nCMD ["python", "/app/app/main.py"]\n',
}

DEFECTIVE_KEY = [
    {"rule_id": "SEC-001", "file": "app/config.py"},
    {"rule_id": "SEC-002", "file": ".env"},
    {"rule_id": "SEC-003", "file": ".github/workflows/ci.yml"},
    {"rule_id": "SEC-004", "file": ".github/workflows/greet.yml"},
    {"rule_id": "SEC-005", "file": ".github/workflows/greet.yml"},
    {"rule_id": "SEC-006", "file": "app/main.py"},
    {"rule_id": "HYG-002", "file": ".gitignore"},
    {"rule_id": "HYG-003", "file": "app/__pycache__/main.cpython-311.pyc"},
    {"rule_id": "DEP-002", "file": "requirements.txt"},
    {"rule_id": "DEP-003", "file": "setup.py"},
    {"rule_id": "DEP-005", "file": "app/main.py"},
    {"rule_id": "CQ-002", "file": "app/main.py"},
    {"rule_id": "TEST-003", "file": "tests/test_main.py"},
    {"rule_id": "CI-002", "file": ".github/workflows/ci.yml"},
    {"rule_id": "CI-003", "file": ".github/workflows/ci.yml"},
    {"rule_id": "CI-005", "file": ".github/workflows/greet.yml"},
    {"rule_id": "CFG-001", "file": "app/config.py"},
    {"rule_id": "CFG-002", "file": ".vscode/settings.json"},
    {"rule_id": "CFG-003", "file": "app/config.py"},
    {"rule_id": "BLD-003", "file": "Dockerfile"},
]

HEALTHY = {
    "README.md": "# tinycalc\n\nA tiny calculator library.\n\n## Development\n\n```\npip install -e .[dev]\npytest\n```\n",
    ".gitignore": "__pycache__/\n*.py[cod]\n.venv/\n.env\ndist/\nbuild/\n*.egg-info/\n.pytest_cache/\n.ruff_cache/\n",
    "pyproject.toml": (
        "[project]\nname = 'tinycalc'\nversion = '1.2.0'\nrequires-python = '>=3.10'\n"
        "dependencies = ['click>=8.1,<9']\n\n[project.optional-dependencies]\ndev = ['pytest>=8', 'ruff>=0.5']\n\n"
        "[tool.ruff]\nline-length = 100\n"
    ),
    "src/tinycalc/__init__.py": "from .ops import add, divide\n\n__all__ = ['add', 'divide']\n",
    "src/tinycalc/ops.py": (
        "def add(a: float, b: float) -> float:\n    return a + b\n\n\n"
        "def divide(a: float, b: float) -> float:\n    if b == 0:\n        raise ZeroDivisionError('b must be non-zero')\n    return a / b\n"
    ),
    "tests/test_ops.py": (
        "import pytest\n\nfrom tinycalc import add, divide\n\n\n"
        "def test_add():\n    assert add(2, 3) == 5\n\n\n"
        "def test_divide_by_zero():\n    with pytest.raises(ZeroDivisionError):\n        divide(1, 0)\n"
    ),
    ".env.example": "API_TOKEN=your-token-here\n",
    ".github/workflows/ci.yml": (
        "name: CI\n"
        "on:\n  push:\n    branches: [main]\n  pull_request:\n"
        "permissions:\n  contents: read\n"
        "jobs:\n"
        "  test:\n"
        "    runs-on: ubuntu-latest\n"
        "    strategy:\n      matrix:\n        python-version: ['3.10', '3.12']\n"
        "    steps:\n"
        "      - uses: actions/checkout@v4\n"
        "      - uses: actions/setup-python@v5\n"
        "        with:\n          python-version: ${{ matrix.python-version }}\n"
        "      - run: pip install -e .[dev]\n"
        "      - run: ruff check .\n"
        "      - run: pytest\n"
        "      - run: pip install build && python -m build\n"
    ),
}


def make_repo(root: Path, files: dict[str, str], branch: str) -> None:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    git = ["git", "-c", "user.name=Demo Dev", "-c", "user.email=dev@example.com"]
    subprocess.run([*git, "init", "-q", "-b", branch], cwd=root, check=True)
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    subprocess.run([*git, "add", "-A", "-f"], cwd=root, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "Initial commit"], cwd=root, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/demo-repos")
    out = Path(ap.parse_args().out)
    make_repo(out / "defective", DEFECTIVE, "main")
    make_repo(out / "healthy", HEALTHY, "main")
    (out / "defective.key.json").write_text(json.dumps(DEFECTIVE_KEY, indent=2))
    (out / "healthy.key.json").write_text("[]\n")
    print(f"Created {out / 'defective'} ({len(DEFECTIVE_KEY)} known defects) and {out / 'healthy'} (0 known defects)")


if __name__ == "__main__":
    main()
