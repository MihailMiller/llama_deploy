import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from llama_deploy.cli import _detect_auth_mode
from llama_deploy.config import AuthMode


REPO_ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def test_detect_auth_mode_from_hash_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            secrets = base / "secrets"
            secrets.mkdir(parents=True, exist_ok=True)
            (secrets / "token_hashes.json").write_text('{"hashes": []}\n', encoding="utf-8")
            self.assertEqual(_detect_auth_mode(base), AuthMode.HASHED)

    def test_detect_auth_mode_from_key_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            secrets = base / "secrets"
            secrets.mkdir(parents=True, exist_ok=True)
            (secrets / "api_keys").write_text("sk-abc\n", encoding="utf-8")
            self.assertEqual(_detect_auth_mode(base), AuthMode.PLAINTEXT)

    def test_batch_help_prints_batch_flags(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-m", "llama_deploy", "deploy", "--batch", "--help"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("--base-dir", proc.stdout)
        self.assertIn("--auth-mode", proc.stdout)

    def test_tokens_help_exits_cleanly(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-m", "llama_deploy", "tokens", "--help"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("Manage API tokens", proc.stdout)


if __name__ == "__main__":
    unittest.main()
