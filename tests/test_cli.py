import subprocess
import sys


def test_help_exposes_only_supported_commands():
    completed = subprocess.run(
        [sys.executable, "-m", "smallbatch.cli", "--help"],
        text=True,
        capture_output=True,
        check=True,
    )
    for command in ("init", "doctor", "label", "compile", "select", "run", "status"):
        assert command in completed.stdout
    for removed in ("review", "sweep", "export", "serve", "push"):
        assert removed not in completed.stdout
