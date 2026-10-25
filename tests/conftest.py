import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FAKE_KEY = "AKIA" + "Z3XQ9W7LMN2B4VCD"


def make_git_repo(path: Path, files: dict[str, str]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    git = ["git", "-c", "user.name=Test", "-c", "user.email=t@example.com"]
    subprocess.run([*git, "init", "-q", "-b", "main"], cwd=path, check=True)
    for rel, content in files.items():
        p = path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    subprocess.run([*git, "add", "-A", "-f"], cwd=path, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "init"], cwd=path, check=True)
    return path


SAMPLE_FILES = {
    "README.md": "# Demo\n",
    ".gitignore": "*.log\n",
    "requirements.txt": "flask\nrequests==2.31.0\n",
    "app/config.py": f"import os\nDEBUG = True\nAWS_KEY = '{FAKE_KEY}'\nPATH = '/Users/alice/data'\nTOKEN = os.getenv('TOKEN')\n",
    "app/main.py": "import requests\n\n\ndef f():\n    try:\n        return requests.get('x', verify=False)\n    except:\n        pass\n",
    "app/__pycache__/main.cpython-311.pyc": "junk",
    "debug.log": "log line\n",
    ".env.example": "API_KEY=your-key-here\n",
    ".github/workflows/ci.yml": (
        "name: CI\non:\n  push:\n    branches: [main]\npermissions: write-all\njobs:\n  test:\n    runs-on: ubuntu-latest\n"
        "    steps:\n      - uses: actions/checkout@v4\n      - uses: other/action@master\n      - run: pytest || true\n"
        "        continue-on-error: true\n"
    ),
}


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    return make_git_repo(tmp_path / "repo", SAMPLE_FILES)


LLM_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY",
    "LLM_PROVIDER",
    "LLM_MODEL",
    "OPENAI_BASE_URL",
    "LLM_MIN_SECONDS_BETWEEN_CALLS",
    "OLLAMA_HOST",
    "OLLAMA_NUM_CTX",
)


@pytest.fixture(autouse=True)
def isolated_llm_environment(monkeypatch):
    """No test may depend on, or load, the developer's real provider keys or .env file."""
    for var in LLM_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
