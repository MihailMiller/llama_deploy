import tempfile
import unittest
from pathlib import Path
from unittest import mock

from llama_deploy.config import BackendKind, Config, ModelSpec, NetworkConfig
from llama_deploy.orchestrator import _auto_optimize_cfg


def _cfg(base_dir: Path, *, models_max: int = 1) -> Config:
    return Config(
        base_dir=base_dir,
        backend=BackendKind.CPU,
        network=NetworkConfig(bind_host="127.0.0.1", port=8080, publish=True),
        swap_gib=8,
        models_max=models_max,
        parallel=1,
        api_token=None,
        api_token_name="default",
        hf_token=None,
        skip_download=False,
        llm=ModelSpec(
            hf_repo="Qwen/Qwen3-8B-GGUF",
            candidate_patterns=["Q4_K_M", "Q5_K_M"],
            ctx_len=3072,
        ),
        emb=ModelSpec(
            hf_repo="Qwen/Qwen3-Embedding-0.6B-GGUF",
            candidate_patterns=["Q8_0"],
            ctx_len=2048,
            is_embedding=True,
        ),
    )


class OrchestratorAutoOptimizeTests(unittest.TestCase):
    def test_auto_optimize_keeps_models_max_at_least_two(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = _cfg(Path(td), models_max=1)
            with mock.patch("llama_deploy.orchestrator._detect_mem_total_gib", return_value=8.0), \
                 mock.patch("llama_deploy.log.log_line"):
                tuned = _auto_optimize_cfg(cfg)
            self.assertGreaterEqual(tuned.models_max, 2)


if __name__ == "__main__":
    unittest.main()
