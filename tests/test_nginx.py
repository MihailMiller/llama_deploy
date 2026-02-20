import tempfile
import unittest
from pathlib import Path
from unittest import mock

from llama_deploy import nginx


class NginxConfigTests(unittest.TestCase):
    def test_write_local_config_with_auth_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sites_available = root / "sites-available"
            sites_enabled = root / "sites-enabled"
            sites_available.mkdir(parents=True, exist_ok=True)
            sites_enabled.mkdir(parents=True, exist_ok=True)

            captured = {}

            def _fake_write_file(path: Path, content: str, *, mode=None) -> None:
                captured["path"] = path
                captured["content"] = content
                captured["mode"] = mode

            with mock.patch.object(nginx, "_SITES_AVAILABLE", sites_available), \
                 mock.patch.object(nginx, "_SITES_ENABLED", sites_enabled), \
                 mock.patch("llama_deploy.system.write_file", side_effect=_fake_write_file), \
                 mock.patch.object(nginx, "sh"), \
                 mock.patch("pathlib.Path.symlink_to", return_value=None):
                nginx.write_nginx_local_config(
                    bind_host="127.0.0.1",
                    port=8080,
                    upstream_port=8081,
                    use_auth_sidecar=True,
                    sidecar_port=9000,
                )

            self.assertIn("auth_request /auth;", captured["content"])
            self.assertIn("proxy_pass         http://127.0.0.1:8081;", captured["content"])


if __name__ == "__main__":
    unittest.main()
