import json
import tempfile
import unittest
from pathlib import Path

from llama_deploy.config import AuthMode
from llama_deploy.tokens import TokenStore


class TokenStoreTests(unittest.TestCase):
    def test_plaintext_mode_persists_explicit_token_value(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            secrets = Path(td) / "secrets"
            store = TokenStore(secrets, auth_mode=AuthMode.PLAINTEXT)

            record = store.create_token("app", value="sk-explicit-token")
            self.assertEqual(record.value, "sk-explicit-token")

            saved = json.loads((secrets / "tokens.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["tokens"][0]["value"], "sk-explicit-token")

            keyfile = (secrets / "api_keys").read_text(encoding="utf-8")
            self.assertIn("sk-explicit-token", keyfile)

    def test_hashed_mode_never_persists_plaintext(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            secrets = Path(td) / "secrets"
            store = TokenStore(secrets, auth_mode=AuthMode.HASHED)

            record = store.create_token("app", value="sk-secret-token")
            self.assertEqual(record.value, "sk-secret-token")

            saved = json.loads((secrets / "tokens.json").read_text(encoding="utf-8"))
            token = saved["tokens"][0]
            self.assertIsNone(token["value"])
            self.assertTrue(token["hash"])

            hashes = json.loads((secrets / "token_hashes.json").read_text(encoding="utf-8"))
            self.assertEqual(len(hashes["hashes"]), 1)

            with self.assertRaises(ValueError):
                store.show_token(record.id)


if __name__ == "__main__":
    unittest.main()
