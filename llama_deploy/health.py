"""
Health polling and smoke tests.

curl_smoke_tests receives ModelSpec objects and uses effective_alias for model
names in request bodies — no hardcoded "Qwen/Qwen3-8B" strings.
"""

from __future__ import annotations

import json
import time
import urllib.request

from llama_deploy.config import ModelSpec
from llama_deploy.log import die, log_line, sh


def wait_health(url: str, timeout_s: int = 300) -> None:
    from tqdm import tqdm

    deadline = time.time() + timeout_s
    with tqdm(total=timeout_s, desc="Waiting for /health", unit="s") as bar:
        last = time.time()
        while time.time() < deadline:
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=3) as resp:
                    if resp.status == 200:
                        tqdm.write("[OK] /health returned 200")
                        return
            except Exception:
                pass
            now = time.time()
            step = int(now - last)
            if step > 0:
                bar.update(step)
                last = now
            time.sleep(1)
    die(f"Service did not become healthy within {timeout_s}s (check: docker logs llama-router).")


def curl_smoke_tests(
    base_url: str,
    token: str,
    llm: ModelSpec,
    emb: ModelSpec,
) -> None:
    """
    Run three smoke tests against the OpenAI-compatible API.

    Model names come from spec.effective_alias so they match what the server
    advertises in /v1/models — regardless of which HF repo was used.
    Fixes Bug 2 (hardcoded Qwen model names in the original script).
    """
    log_line(f"[SMOKE] Starting smoke tests against {base_url}")

    # 1. Model listing
    sh(
        f'curl -fsS "{base_url}/v1/models" '
        f'-H "Authorization: Bearer {token}" | head -c 800'
    )

    # 2. Embeddings
    emb_payload = json.dumps({"model": emb.effective_alias, "input": ["hello world"]})
    sh(
        f'curl -fsS "{base_url}/v1/embeddings" '
        f'-H "Authorization: Bearer {token}" '
        f'-H "Content-Type: application/json" '
        f"-d '{emb_payload}' | head -c 800"
    )

    # 3. Chat completion
    chat_payload = json.dumps({
        "model": llm.effective_alias,
        "messages": [{"role": "user", "content": "Say hello in 5 words."}],
        "max_tokens": 64,
        "temperature": 0.2,
    })
    sh(
        f'curl -fsS "{base_url}/v1/chat/completions" '
        f'-H "Authorization: Bearer {token}" '
        f'-H "Content-Type: application/json" '
        f"-d '{chat_payload}' | head -c 800"
    )


def sanity_checks(cfg) -> None:
    """Docker status, recent logs, and port-binding verification."""
    sh(
        "docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}' "
        "| sed -n '1,30p'",
        check=False,
    )
    sh("docker logs --tail 200 llama-router || true", check=False)
    sh("ss -lntp | sed -n '1,200p'", check=False)

    net = cfg.network
    if net.publish and not net.is_public:
        # Verify no accidental 0.0.0.0 exposure when we asked for loopback only
        sh(
            f"ss -lntp | grep -E '0\\.0\\.0\\.0:{net.port}\\b|\\[::\\]:{net.port}\\b' "
            f"&& exit 1 || true",
            check=False,
        )
