"""
Logging, redaction, and subprocess execution.

This module has no imports from other llama_deploy modules so it can be
imported by everything else without creating circular dependencies.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

LOG_PATH = Path("/var/log/llamacpp_deploy.log")

# Redact tokens from on-screen output and log
REDACT_PATTERNS = [
    re.compile(r"(Authorization:\s*Bearer\s+)[^\s\"']+", re.IGNORECASE),
    re.compile(r"(x-api-key:\s*)[^\s\"']+", re.IGNORECASE),
    re.compile(r"(HF_TOKEN=)[^\s]+"),
    re.compile(r"(--hf-token\s+)[^\s]+"),
]


def redact(s: str) -> str:
    out = s
    for pat in REDACT_PATTERNS:
        out = pat.sub(r"\1<REDACTED>", out)
    return out


def log_line(s: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(s + "\n")


def die(msg: str, code: int = 1) -> None:
    # Import tqdm lazily to avoid a hard boot-time dependency
    try:
        from tqdm import tqdm
        tqdm.write(f"[FATAL] {msg}")
    except Exception:
        print(f"[FATAL] {msg}", file=sys.stderr, flush=True)
    log_line(f"[FATAL] {msg}")
    sys.exit(code)


def sh(
    cmd: str,
    *,
    check: bool = True,
    env: Optional[Dict[str, str]] = None,
) -> int:
    """Run a bash command, stream stdout/stderr, log everything (redacted)."""
    from tqdm import tqdm  # imported here; tqdm is guaranteed present by orchestrator boot

    safe_cmd = redact(cmd)
    tqdm.write(f"\n$ {safe_cmd}")
    log_line(f"\n$ {safe_cmd}")

    proc = subprocess.Popen(
        ["bash", "-lc", cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        preexec_fn=os.setsid,  # kill whole process group on Ctrl-C
    )

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            tqdm.write(line)
            log_line(redact(line))
    except KeyboardInterrupt:
        tqdm.write("[WARN] Ctrl-C received. Terminating command...")
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            pass
        raise

    rc = proc.wait()
    if check and rc != 0:
        die(f"Command failed (exit {rc}): {safe_cmd}")
    return rc
