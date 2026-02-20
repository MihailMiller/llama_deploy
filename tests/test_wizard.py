import unittest
from unittest import mock

from llama_deploy.config import AccessProfile, AuthMode, NetworkConfig
from llama_deploy.wizard import _commit_local_hashed_proxy_port


class WizardPortCommitTests(unittest.TestCase):
    def test_noop_when_not_hashed(self) -> None:
        network = NetworkConfig(
            bind_host="127.0.0.1",
            port=8080,
            publish=True,
            open_firewall=False,
            configure_ufw=True,
            access_profile=AccessProfile.LOCALHOST,
            lan_cidr=None,
        )
        result = _commit_local_hashed_proxy_port(
            network=network,
            auth_mode=AuthMode.PLAINTEXT,
            domain=None,
        )
        self.assertEqual(result.port, 8080)

    def test_commit_existing_free_port(self) -> None:
        network = NetworkConfig(
            bind_host="127.0.0.1",
            port=8080,
            publish=True,
            open_firewall=False,
            configure_ufw=True,
            access_profile=AccessProfile.LOCALHOST,
            lan_cidr=None,
        )

        with mock.patch("llama_deploy.wizard._section"), \
             mock.patch("llama_deploy.wizard._info"), \
             mock.patch("llama_deploy.wizard._warn"), \
             mock.patch("llama_deploy.wizard._is_bind_port_free", return_value=True), \
             mock.patch("llama_deploy.wizard._confirm", return_value=True), \
             mock.patch("llama_deploy.wizard._prompt_int") as prompt_int:
            result = _commit_local_hashed_proxy_port(
                network=network,
                auth_mode=AuthMode.HASHED,
                domain=None,
            )

        self.assertEqual(result.port, 8080)
        prompt_int.assert_not_called()

    def test_reprompt_until_user_commits_free_port(self) -> None:
        network = NetworkConfig(
            bind_host="127.0.0.1",
            port=8080,
            publish=True,
            open_firewall=False,
            configure_ufw=True,
            access_profile=AccessProfile.LOCALHOST,
            lan_cidr=None,
        )

        with mock.patch("llama_deploy.wizard._section"), \
             mock.patch("llama_deploy.wizard._info"), \
             mock.patch("llama_deploy.wizard._warn"), \
             mock.patch("llama_deploy.wizard._is_bind_port_free", side_effect=[False, True]), \
             mock.patch("llama_deploy.wizard._confirm", return_value=True), \
             mock.patch("llama_deploy.wizard._prompt_int", return_value=18080):
            result = _commit_local_hashed_proxy_port(
                network=network,
                auth_mode=AuthMode.HASHED,
                domain=None,
            )

        self.assertEqual(result.port, 18080)


if __name__ == "__main__":
    unittest.main()
