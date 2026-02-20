import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from llama_deploy.config import AuthMode
from llama_deploy.orchestrator import _ensure_first_token
from llama_deploy.tokens import TokenStore


class OrchestratorTokenTests(unittest.TestCase):
    def test_first_token_honors_explicit_api_token(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            secrets = base / "secrets"
            cfg = SimpleNamespace(
                secrets_dir=secrets,
                auth_mode=AuthMode.PLAINTEXT,
                api_token_name="my-token",
                api_token="sk-my-explicit-token",
            )

            runtime = _ensure_first_token(cfg)
            self.assertEqual(runtime.value, "sk-my-explicit-token")
            self.assertIsNone(runtime.temporary_id)

            store = TokenStore(secrets, auth_mode=AuthMode.PLAINTEXT)
            active = store.active_tokens()
            self.assertEqual(len(active), 1)
            self.assertEqual(active[0].value, "sk-my-explicit-token")

    def test_hashed_redeploy_creates_temporary_smoke_token(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            secrets = base / "secrets"
            store = TokenStore(secrets, auth_mode=AuthMode.HASHED)
            first = store.create_token("existing")

            cfg = SimpleNamespace(
                secrets_dir=secrets,
                auth_mode=AuthMode.HASHED,
                api_token_name="ignored",
                api_token=None,
            )

            runtime = _ensure_first_token(cfg)
            self.assertTrue(runtime.value.startswith("sk-"))
            self.assertNotEqual(runtime.value, first.value)
            self.assertIsNotNone(runtime.temporary_id)


if __name__ == "__main__":
    unittest.main()
