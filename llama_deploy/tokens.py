"""
API token persistence and lifecycle management.

Storage layout under <base_dir>/secrets/:

  Plaintext mode (default):
    tokens.json     — structured metadata; 'value' field holds plaintext token
    api_keys        — flat one-token-per-line file read by llama-server

  Hashed mode (--auth-mode hashed):
    tokens.json     — structured metadata; 'value' is null, 'hash' holds SHA-256
    token_hashes.json — flat list of active SHA-256 hashes read by the auth sidecar
    api_keys        — not created

In hashed mode the plaintext token value is returned to the caller at creation
time and never written to disk. Subsequent calls to show_token() will raise
ValueError because the plaintext is unrecoverable.

Token IDs use the prefix "tk_" + 12 hex chars.
Token values use the prefix "sk-" + 48 url-safe chars.

This module imports only from log.py (and stdlib) to stay low in the
dependency graph. The orchestrator and CLI both use TokenStore directly.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import secrets
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

from llama_deploy.config import AuthMode


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TokenRecord:
    id: str            # "tk_<12 hex chars>"
    name: str          # human label, e.g. "my-app"
    created_at: str    # ISO 8601 UTC
    revoked: bool = False
    revoked_at: Optional[str] = None
    value: Optional[str] = None  # plaintext; set in plaintext mode, None in hashed mode
    hash: Optional[str] = None   # SHA-256 hex; set in hashed mode, None in plaintext mode

    @classmethod
    def _from_dict(cls, d: dict) -> "TokenRecord":
        return cls(
            id=d["id"],
            name=d["name"],
            created_at=d["created_at"],
            revoked=d.get("revoked", False),
            revoked_at=d.get("revoked_at"),
            value=d.get("value"),
            hash=d.get("hash"),
        )

    def _to_dict(self) -> dict:
        return asdict(self)

    @property
    def status(self) -> str:
        return "REVOKED" if self.revoked else "active"


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class TokenStore:
    """
    Manages the token metadata JSON file and the llama-server / sidecar keyfile.

    All mutating methods call the appropriate sync method so the keyfile stays
    consistent with the active token set.
    """

    def __init__(self, secrets_dir: Path, auth_mode: AuthMode = AuthMode.PLAINTEXT) -> None:
        self._dir        = secrets_dir
        self._auth_mode  = auth_mode
        self._json_path  = secrets_dir / "tokens.json"
        self._keyfile    = secrets_dir / "api_keys"           # plaintext mode
        self._hashfile   = secrets_dir / "token_hashes.json"  # hashed mode

    @property
    def auth_mode(self) -> AuthMode:
        return self._auth_mode

    # -----------------------------------------------------------------------
    # Read
    # -----------------------------------------------------------------------

    def list_tokens(self) -> List[TokenRecord]:
        """Return all tokens (including revoked), ordered by creation date."""
        return self._load()

    def active_tokens(self) -> List[TokenRecord]:
        """Return only non-revoked tokens."""
        return [t for t in self._load() if not t.revoked]

    def show_token(self, token_id: str) -> TokenRecord:
        """
        Return a single token by id; raises KeyError if not found.
        Raises ValueError in hashed mode (plaintext was never stored).
        """
        for t in self._load():
            if t.id == token_id:
                if self._auth_mode == AuthMode.HASHED and t.value is None:
                    raise ValueError(
                        f"Token '{token_id}' was created in hashed mode — "
                        "the plaintext value was never stored on disk and cannot be recovered."
                    )
                return t
        raise KeyError(f"Token not found: {token_id}")

    # -----------------------------------------------------------------------
    # Write
    # -----------------------------------------------------------------------

    def create_token(self, name: str, value: Optional[str] = None) -> TokenRecord:
        """
        Generate a new token, persist it, and resync the keyfile.

        In plaintext mode: stores the raw value in tokens.json and api_keys.
        In hashed mode: stores only the SHA-256 hash; the caller receives the
        record with value set (show it once) but tokens.json holds value=None.
        If value is provided, it is used as the token plaintext instead of
        generating a random token.
        """
        token_id  = "tk_" + secrets.token_hex(6)         # 12 hex chars
        raw_value = value or ("sk-" + secrets.token_urlsafe(48))
        if not raw_value.strip():
            raise ValueError("Token value must not be empty.")
        now       = dt.datetime.now(dt.timezone.utc).isoformat()

        if self._auth_mode == AuthMode.HASHED:
            token_hash = hashlib.sha256(raw_value.encode()).hexdigest()
            # Store hash only — value is NOT persisted
            stored_record = TokenRecord(
                id=token_id, name=name, created_at=now,
                value=None, hash=token_hash,
            )
            # Return record with value populated so the caller can show it once
            display_record = TokenRecord(
                id=token_id, name=name, created_at=now,
                value=raw_value, hash=token_hash,
            )
        else:
            stored_record = TokenRecord(
                id=token_id, name=name, created_at=now, value=raw_value,
            )
            display_record = stored_record

        tokens = self._load()
        tokens.append(stored_record)
        self._save(tokens)
        self._sync(tokens)
        return display_record

    def revoke_token(self, token_id: str) -> TokenRecord:
        """
        Mark a token as revoked and resync the keyfile.
        Returns the updated record; raises KeyError if not found.
        """
        tokens = self._load()
        for t in tokens:
            if t.id == token_id:
                if t.revoked:
                    raise ValueError(f"Token {token_id} is already revoked.")
                t.revoked    = True
                t.revoked_at = dt.datetime.now(dt.timezone.utc).isoformat()
                self._save(tokens)
                self._sync(tokens)
                return t
        raise KeyError(f"Token not found: {token_id}")

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _load(self) -> List[TokenRecord]:
        if not self._json_path.exists():
            return []
        raw = json.loads(self._json_path.read_text(encoding="utf-8"))
        return [TokenRecord._from_dict(d) for d in raw.get("tokens", [])]

    def _save(self, tokens: List[TokenRecord]) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        data = json.dumps({"tokens": [t._to_dict() for t in tokens]}, indent=2)
        self._json_path.write_text(data + "\n", encoding="utf-8")
        os.chmod(self._json_path, 0o600)

    def _sync(self, tokens: Optional[List[TokenRecord]] = None) -> None:
        """Dispatch to the correct keyfile writer based on auth mode."""
        if tokens is None:
            tokens = self._load()
        if self._auth_mode == AuthMode.HASHED:
            self._sync_hashfile(tokens)
        else:
            self._sync_keyfile(tokens)

    def _sync_keyfile(self, tokens: List[TokenRecord]) -> None:
        """
        Rewrite api_keys with one plaintext value per line (active tokens only).
        llama-server reads this file; revoked tokens are simply absent.
        """
        active = [t.value for t in tokens if not t.revoked and t.value]
        self._keyfile.write_text("\n".join(active) + "\n", encoding="utf-8")
        os.chmod(self._keyfile, 0o600)

    def _sync_hashfile(self, tokens: List[TokenRecord]) -> None:
        """
        Rewrite token_hashes.json with active SHA-256 hashes.
        The auth sidecar reads this file; revoked tokens are simply absent.
        Reload on every request in the sidecar means revocation is instant.
        """
        active_hashes = [t.hash for t in tokens if not t.revoked and t.hash]
        data = json.dumps({"hashes": active_hashes}, indent=2)
        self._hashfile.write_text(data + "\n", encoding="utf-8")
        os.chmod(self._hashfile, 0o600)

    def restart_hint(self) -> str:
        """Returns a one-liner the user can run to pick up keyfile changes."""
        return "docker compose -f <base_dir>/docker-compose.yml restart llama"
