import unittest
from unittest import mock

from llama_deploy.system import pick_free_bind_port


class SystemPortSelectionTests(unittest.TestCase):
    def test_pick_free_bind_port_keeps_preferred_when_available(self) -> None:
        with mock.patch("llama_deploy.system._is_bind_port_free", return_value=True):
            chosen = pick_free_bind_port("0.0.0.0", 3000)
        self.assertEqual(chosen, 3000)

    def test_pick_free_bind_port_moves_to_next_available(self) -> None:
        availability = {3000: False, 3001: False, 3002: True}

        def _free(_host: str, port: int) -> bool:
            return availability.get(port, False)

        with mock.patch("llama_deploy.system._is_bind_port_free", side_effect=_free):
            chosen = pick_free_bind_port("0.0.0.0", 3000)
        self.assertEqual(chosen, 3002)

    def test_pick_free_bind_port_respects_avoid_set(self) -> None:
        availability = {3000: True, 3001: True, 3002: True}

        def _free(_host: str, port: int) -> bool:
            return availability.get(port, False)

        with mock.patch("llama_deploy.system._is_bind_port_free", side_effect=_free):
            chosen = pick_free_bind_port("127.0.0.1", 3000, avoid={3000, 3001})
        self.assertEqual(chosen, 3002)


if __name__ == "__main__":
    unittest.main()
