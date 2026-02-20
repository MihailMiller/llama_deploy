"""
Deployment orchestrator: Step protocol, run_steps runner, and run_deploy().

The Step dataclass replaces the original (label, fn) tuple list that used
string-matching to capture return values (Bug 4). Every step's return value
is stored in step.result and can be read by subsequent steps without re-calling
the function (Bug 1).

Token management uses TokenStore (tokens.py) instead of the old ensure_token_file.
The token value is shown ONCE after deployment if it was newly created.

All steps â€” including HF download, config writing, and Docker startup â€” are
included in a single all_steps list so the progress bar reflects the true
total (Bug 7).

Auth modes
----------
Plaintext (default):
  llama-server uses --api-key-file; tokens stored as plaintext in api_keys.

Hashed:
  NGINX auth_request delegates to the llama-auth sidecar; only SHA-256
  hashes are stored in token_hashes.json; plaintext never touches disk.
  - If cfg.use_tls: the TLS step also configures auth_request in NGINX.
  - Otherwise: a dedicated local_proxy_step installs NGINX without certbot.
"""

from __future__ import annotations

import datetime as dt
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional


# ---------------------------------------------------------------------------
# Progress helper (tqdm with a graceful fallback)
# ---------------------------------------------------------------------------

class _TqdmStub:
    """Fallback progress helper when tqdm is not available yet."""

    def __init__(self, total: int = 0, desc: str = "", unit: str = "") -> None:
        self.total = total
        self.desc = desc
        self.unit = unit

    @staticmethod
    def write(msg: str) -> None:
        print(msg, flush=True)

    def update(self, _n: int = 1) -> None:
        return

    def __enter__(self) -> "_TqdmStub":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> bool:
        return False


tqdm = _TqdmStub


def _ensure_tqdm(allow_install: bool = True) -> None:
    """
    Ensure tqdm is importable. On non-root import paths we skip package install
    attempts and fall back to a minimal stub.
    """
    global tqdm

    try:
        from tqdm import tqdm as real_tqdm
        tqdm = real_tqdm
        return
    except Exception:
        pass

    if not allow_install:
        return

    print("[BOOT] Installing tqdm (python3-tqdm)...", flush=True)
    subprocess.run(
        ["bash", "-lc", "apt-get update -y && apt-get install -y python3-tqdm"],
        check=False,
    )
    try:
        from tqdm import tqdm as real_tqdm
        tqdm = real_tqdm
        return
    except Exception:
        print("[BOOT] apt install failed; trying pip...", flush=True)
        subprocess.run(
            ["bash", "-lc", "apt-get update -y && apt-get install -y python3-pip"],
            check=False,
        )
        subprocess.run(
            ["bash", "-lc", "python3 -m pip install --no-cache-dir -U pip tqdm"],
            check=False,
        )
        try:
            from tqdm import tqdm as real_tqdm
            tqdm = real_tqdm
            return
        except Exception as e:
            print(f"[WARN] tqdm unavailable ({e}); continuing without progress bar.", flush=True)
            tqdm = _TqdmStub


_ensure_tqdm(allow_install=False)


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

@dataclass
class Step:
    """
    One idempotent deployment step.

    result is populated by run_steps() after fn() returns. Downstream steps
    close over the Step object and read step.result rather than re-calling fn().
    """

    label:   str
    fn:      Callable[[], Any]
    skip_if: Optional[Callable[[], bool]] = None
    result:  Any = field(default=None, repr=False)


@dataclass
class TokenRuntime:
    """
    Runtime token material used by deployment validation and summary printing.

    temporary_id is set only when a temporary validation token was created
    (hashed mode re-deploy where existing tokens have no recoverable plaintext).
    """

    value: str
    temporary_id: Optional[str] = None


def run_steps(steps: List[Step], bar: tqdm) -> None:
    from llama_deploy.log import log_line
    import time

    for step in steps:
        if step.skip_if and step.skip_if():
            tqdm.write(f"[SKIP] {step.label}")
            log_line(f"[SKIP] {step.label}")
            bar.update(1)
            continue

        tqdm.write(f"\n[STEP] {step.label}")
        log_line(f"[STEP] {step.label}")
        t0 = time.time()
        step.result = step.fn()
        elapsed = time.time() - t0
        tqdm.write(f"[DONE] {step.label} ({elapsed:.1f}s)")
        log_line(f"[DONE] {step.label} ({elapsed:.1f}s)")
        bar.update(1)


# ---------------------------------------------------------------------------
# Token helper
# ---------------------------------------------------------------------------

def _ensure_first_token(cfg) -> TokenRuntime:
    """
    Return token material for deployment validation.

    If no token exists yet, create the first token (honoring cfg.api_token when
    provided). In hashed mode on re-deploy, existing token plaintext is not
    recoverable; create a temporary token for smoke tests and revoke it after
    deployment.
    """
    from llama_deploy.tokens import TokenStore

    store = TokenStore(cfg.secrets_dir, auth_mode=cfg.auth_mode)
    active = store.active_tokens()
    if active:
        tqdm.write(f"[OK] Existing token found in store: {active[0].id}")
        if active[0].value:
            return TokenRuntime(value=active[0].value)

        smoke = store.create_token("__deploy-smoke-test")
        tqdm.write(f"[OK] Created temporary smoke-test token ({smoke.id}) for validation.")
        return TokenRuntime(value=smoke.value, temporary_id=smoke.id)

    record = store.create_token(cfg.api_token_name, value=cfg.api_token)
    tqdm.write(f"[OK] Created token '{record.name}' ({record.id})")
    return TokenRuntime(value=record.value)


def _is_new_token(cfg) -> bool:
    """Return True if no tokens exist yet (first-run detection for post-deploy message)."""
    from llama_deploy.tokens import TokenStore
    return len(TokenStore(cfg.secrets_dir, auth_mode=cfg.auth_mode).active_tokens()) == 0


# ---------------------------------------------------------------------------
# run_deploy()  â€” the core deployment pipeline
# ---------------------------------------------------------------------------

def run_deploy(cfg) -> None:
    """
    Execute all deployment steps for the given Config.

    Called by both wizard mode (cli._run_deploy_wizard) and batch mode
    (cli.dispatch with --batch flag).
    """
    from llama_deploy.config import AuthMode
    from llama_deploy.system import (
        require_root_reexec,
        detect_ubuntu,
        ensure_base_packages,
        ensure_unattended_upgrades,
        ensure_docker,
        ensure_swap,
        ensure_firewall,
    )
    from llama_deploy.model import resolve_model
    from llama_deploy.tokens import TokenStore
    from llama_deploy.service import (
        write_models_ini,
        write_compose,
        write_auth_sidecar_script,
        docker_pull,
        docker_compose_up,
    )
    from llama_deploy.health import wait_health, curl_smoke_tests, sanity_checks
    from llama_deploy.log import log_line, redact, LOG_PATH

    require_root_reexec()
    detect_ubuntu()
    _ensure_tqdm(allow_install=True)

    tqdm.write(f"[INFO] Logging to: {LOG_PATH}")
    log_line(f"=== START {dt.datetime.now(dt.timezone.utc).isoformat()}Z ===")
    log_line("Args: " + redact(" ".join(sys.argv)))

    # Ensure directories exist before any step runs
    for d in (cfg.models_dir, cfg.presets_dir, cfg.cache_dir, cfg.secrets_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Detect first-run before the token step runs (for post-deploy message)
    first_run = _is_new_token(cfg)

    # -----------------------------------------------------------------------
    # Phase 1: System preparation
    # -----------------------------------------------------------------------
    token_step = Step(
        label="Create API token",
        fn=lambda: _ensure_first_token(cfg),
    )

    system_steps: List[Step] = [
        Step("Install base packages",      ensure_base_packages),
        Step("Enable unattended upgrades", ensure_unattended_upgrades),
        Step("Install / enable Docker",    ensure_docker),
        Step("Ensure swap",                lambda: ensure_swap(cfg.swap_gib)),
        Step(
            "Firewall (UFW) hardening",
            fn=lambda: ensure_firewall(cfg.network),
            skip_if=lambda: not cfg.network.configure_ufw,
        ),
        token_step,
    ]

    # -----------------------------------------------------------------------
    # Phase 2: Model resolution and download
    # -----------------------------------------------------------------------
    llm_step = Step(
        label="Resolve + download LLM GGUF",
        fn=lambda: resolve_model(cfg.llm, cfg.models_dir, cfg.hf_token),
        skip_if=lambda: cfg.skip_download,
    )
    emb_step = Step(
        label="Resolve + download embedding GGUF",
        fn=lambda: resolve_model(cfg.emb, cfg.models_dir, cfg.hf_token),
        skip_if=lambda: cfg.skip_download,
    )

    def _resolve_from_disk() -> None:
        """When --skip-download, fill step.result from whatever .gguf is on disk."""
        if not cfg.skip_download:
            return
        from llama_deploy.model import sha256_file
        from llama_deploy.log import die

        for step, spec in ((llm_step, cfg.llm), (emb_step, cfg.emb)):
            matches = [
                f for f in cfg.models_dir.glob("*.gguf")
                if any(p in f.name for p in spec.candidate_patterns)
            ]
            if not matches:
                die(
                    f"--skip-download requested but no matching GGUF found for "
                    f"{spec.hf_repo} in {cfg.models_dir}. "
                    f"Patterns tried: {spec.candidate_patterns}"
                )
            chosen = matches[0]
            sha = sha256_file(chosen)
            step.result = spec.with_resolved(
                filename=chosen.name, sha256=sha, size=chosen.stat().st_size
            )
            tqdm.write(f"[SKIP-DL] Using existing {chosen.name} (sha={sha[:12]}...)")

    # -----------------------------------------------------------------------
    # Phase 3: Service configuration and startup
    # -----------------------------------------------------------------------
    def _write_config() -> None:
        write_models_ini(cfg.preset_path, llm_step.result, emb_step.result, cfg.parallel)
        write_compose(cfg.compose_path, cfg)
        if cfg.auth_mode == AuthMode.HASHED:
            write_auth_sidecar_script(cfg.base_dir)

    config_step = Step("Write models.ini + docker-compose.yml", _write_config)
    pull_step   = Step("Pull Docker image",    lambda: docker_pull(cfg.image))
    up_step     = Step("Start Docker Compose", lambda: docker_compose_up(cfg.compose_path))

    # -----------------------------------------------------------------------
    # Phase 3b: NGINX â€” local auth proxy (hashed, no domain) OR TLS (domain set)
    # -----------------------------------------------------------------------
    # When hashed + domain: ensure_tls_for_domain handles NGINX (with auth_request).
    # When hashed + no domain: ensure_local_proxy installs NGINX without certbot.
    # Both steps are mutually exclusive via skip_if.

    def _setup_local_proxy() -> None:
        from llama_deploy.nginx import ensure_local_proxy
        ensure_local_proxy(
            bind_host=cfg.network.bind_host,
            port=cfg.network.port,
            upstream_port=cfg.llama_internal_port,
            configure_ufw=cfg.network.configure_ufw,
            use_auth_sidecar=True,
            sidecar_port=cfg.sidecar_port,
        )

    local_proxy_step = Step(
        label="NGINX local auth proxy (hashed mode)",
        fn=_setup_local_proxy,
        skip_if=lambda: cfg.auth_mode != AuthMode.HASHED or cfg.use_tls,
    )

    def _setup_tls() -> None:
        from llama_deploy.nginx import ensure_tls_for_domain
        ensure_tls_for_domain(
            domain=cfg.domain,
            email=cfg.certbot_email,
            upstream_port=(
                cfg.llama_internal_port
                if cfg.auth_mode == AuthMode.HASHED
                else cfg.network.port
            ),
            configure_ufw=cfg.network.configure_ufw,
            use_auth_sidecar=cfg.auth_mode == AuthMode.HASHED,
            sidecar_port=cfg.sidecar_port,
        )

    tls_step = Step(
        label=f"NGINX + Let's Encrypt TLS ({cfg.domain})",
        fn=_setup_tls,
        skip_if=lambda: not cfg.use_tls,
    )

    # -----------------------------------------------------------------------
    # Phase 4: Validation
    # -----------------------------------------------------------------------
    # Health check against the loopback port (always reachable, regardless of TLS)
    loopback_url = cfg.network.base_url

    if cfg.network.publish or cfg.auth_mode == AuthMode.HASHED:
        health_step = Step(
            "Wait for /health",
            lambda: wait_health(f"{loopback_url}/health", timeout_s=300),
        )
        smoke_step = Step(
            "Smoke tests (OpenAI-compatible routes)",
            lambda: curl_smoke_tests(loopback_url, token_step.result.value, llm_step.result, emb_step.result),
        )
    else:
        project  = cfg.base_dir.name
        net_name = f"{project}_llm_internal"

        def _internal_test() -> None:
            from llama_deploy.log import sh
            sh("docker pull curlimages/curl:8.5.0", check=False)
            sh(
                f"docker run --rm --network {net_name} curlimages/curl:8.5.0 "
                f"sh -lc \"curl -fsS http://llama-router:8080/health && "
                f"curl -fsS http://llama-router:8080/v1/models "
                f"-H 'Authorization: Bearer {token_step.result.value}' | head -c 800\""
            )

        health_step = Step("Internal network smoke test", _internal_test)
        smoke_step  = Step("Sanity checks", lambda: sanity_checks(cfg))

    sanity_step = Step("Sanity checks (ports + logs)", lambda: sanity_checks(cfg))

    service_steps = [
        config_step, pull_step, up_step,
        local_proxy_step, tls_step,
        health_step, smoke_step, sanity_step,
    ]

    all_steps: List[Step] = system_steps + [llm_step, emb_step] + service_steps

    # -----------------------------------------------------------------------
    # Execute
    # -----------------------------------------------------------------------
    try:
        with tqdm(total=len(all_steps), desc="Deploying llama.cpp", unit="step") as bar:
            run_steps(system_steps, bar)
            _resolve_from_disk()
            run_steps([llm_step, emb_step], bar)
            run_steps(service_steps, bar)

        # -------------------------------------------------------------------
        # Post-deploy summary
        # -------------------------------------------------------------------
        _print_summary(cfg, token_step.result.value, first_run)
    finally:
        runtime = token_step.result
        if isinstance(runtime, TokenRuntime) and runtime.temporary_id:
            store = TokenStore(cfg.secrets_dir, auth_mode=cfg.auth_mode)
            try:
                store.revoke_token(runtime.temporary_id)
                tqdm.write(f"[CLEANUP] Revoked temporary smoke-test token ({runtime.temporary_id}).")
            except Exception as e:
                tqdm.write(
                    f"[WARN] Failed to revoke temporary smoke-test token "
                    f"({runtime.temporary_id}): {e}"
                )
                log_line(
                    f"[WARN] Failed to revoke temporary smoke-test token "
                    f"({runtime.temporary_id}): {e}"
                )

        log_line(f"=== END {dt.datetime.now(dt.timezone.utc).isoformat()}Z ===")


def _print_summary(cfg, api_token: str, first_run: bool) -> None:
    from llama_deploy.config import AuthMode

    print()
    print("â•”" + "â•" * 52 + "â•—")
    print("â•‘" + "  Deployment complete".ljust(52) + "â•‘")
    print("â•š" + "â•" * 52 + "â•")
    print()

    if first_run:
        print(f'Your API token "{cfg.api_token_name}":')
        print()
        print(f"  {api_token}")
        print()
        if cfg.auth_mode == AuthMode.HASHED:
            print("  âš   This is shown ONCE. The plaintext is NOT stored on disk.")
            print(f"     Hash stored in: {cfg.secrets_dir}/tokens.json  (mode 600)")
        else:
            print("  âš   This is shown ONCE. Stored in:")
            print(f"     {cfg.secrets_dir}/tokens.json  (mode 600)")
    else:
        print("  Token already existed â€” value unchanged.")
        print(f"  See {cfg.secrets_dir}/tokens.json for token details.")

    print()
    print(f"  Endpoint : {cfg.public_base_url}")
    print(f"  Auth mode: {cfg.auth_mode.value}")
    print(f"  Base dir : {cfg.base_dir}")
    print()
    print("Token management:")
    print("  python -m llama_deploy tokens list")
    print("  python -m llama_deploy tokens create --name <name>")
    print("  python -m llama_deploy tokens revoke <id>")
    print()
    print("Client usage:")
    print("  Authorization: Bearer <token>")
    print(f'  POST /v1/chat/completions  model="{cfg.llm.effective_alias}"')
    print(f'  POST /v1/embeddings        model="{cfg.emb.effective_alias}"')
    print()


# ---------------------------------------------------------------------------
# Legacy entry point (direct invocation without subcommands)
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Called by __main__.py when invoked as `python -m llama_deploy`.
    Delegates to cli.dispatch() which handles TTY detection and subcommands.
    """
    from llama_deploy.cli import dispatch
    dispatch()


if __name__ == "__main__":
    main()

