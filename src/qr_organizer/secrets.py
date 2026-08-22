"""Secret resolution and rotation.

Resolution order, highest first:

1. `$CREDENTIALS_DIRECTORY/<name>` -- systemd-creds, decrypted by systemd into
   a private tmpfs for this unit only. The production path.
2. The configured environment variable (`ANTHROPIC_API_KEY`).
3. `<config dir>/.env`, `KEY=value` lines, which must be mode 600. A looser
   mode is refused rather than quietly used -- a world-readable API key is a
   problem you want to hear about.

The plaintext of every secret belongs in a password manager. Neither the
`.cred` file nor the `.env` file is a backup: both are host-specific, and a
systemd-creds blob is sealed to this host's key/TPM and cannot be restored
anywhere else.
"""

from __future__ import annotations

import logging
import os
import shutil
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import paths

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Secret:
    name: str
    value: str
    source: str

    def masked(self) -> str:
        if len(self.value) <= 8:
            return "*" * len(self.value)
        return f"{self.value[:4]}{'*' * 8}{self.value[-4:]}"


def _from_credentials_directory(name: str) -> Secret | None:
    directory = os.environ.get("CREDENTIALS_DIRECTORY")
    if not directory:
        return None
    candidate = Path(directory) / name
    if not candidate.is_file():
        return None
    value = candidate.read_text(encoding="utf-8").strip()
    if not value:
        log.error("credential %s exists at %s but is empty", name, candidate)
        return None
    return Secret(name=name, value=value, source=f"systemd-creds ({candidate})")


def _from_environment(env_var: str) -> Secret | None:
    value = os.environ.get(env_var, "").strip()
    if not value:
        return None
    return Secret(name=env_var, value=value, source=f"environment ${env_var}")


def _from_dotenv(env_var: str) -> Secret | None:
    env_path = paths.config_dir() / ".env"
    if not env_path.is_file():
        return None
    mode = stat.S_IMODE(env_path.stat().st_mode)
    if mode & 0o077:
        log.error(
            "refusing to read %s: mode is %o, expected 600. Run: chmod 600 %s",
            env_path, mode, env_path,
        )
        return None
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() != env_var:
            continue
        value = value.strip().strip("'\"")
        if value:
            return Secret(name=env_var, value=value, source=f"{env_path}")
    return None


def resolve(credential_name: str, env_var: str) -> Secret | None:
    """Find a secret, or return None having logged where it looked."""
    for finder in (
        lambda: _from_credentials_directory(credential_name),
        lambda: _from_environment(env_var),
        lambda: _from_dotenv(env_var),
    ):
        secret = finder()
        if secret is not None:
            log.debug("resolved secret %s from %s", credential_name, secret.source)
            return secret
    log.warning(
        "secret %r not found: looked in $CREDENTIALS_DIRECTORY/%s, $%s, and %s/.env",
        credential_name, credential_name, env_var, paths.config_dir(),
    )
    return None


def describe_sources(credential_name: str, env_var: str) -> list[tuple[str, bool]]:
    """For the status page: which source has this secret, without revealing it."""
    directory = os.environ.get("CREDENTIALS_DIRECTORY")
    env_path = paths.config_dir() / ".env"
    return [
        (
            f"systemd-creds ($CREDENTIALS_DIRECTORY/{credential_name})",
            bool(directory and (Path(directory) / credential_name).is_file()),
        ),
        (f"environment ${env_var}", bool(os.environ.get(env_var, "").strip())),
        (f"{env_path}", _from_dotenv(env_var) is not None),
    ]


class RotationError(RuntimeError):
    """Rotation could not be completed; the old credential is untouched."""


def rotate_systemd_credential(
    *,
    credential_name: str,
    new_value: str,
    credentials_dir: Path,
    unit: str,
    timeout: int = 60,
) -> str:
    """Re-encrypt a credential and restart the unit, via two narrow sudo rules.

    The web process itself stays unprivileged. `deploy/qr-organizer.sudoers`
    grants exactly these two commands and nothing else. On success the previous
    `.cred` file is deleted immediately.
    """
    if not new_value.strip():
        raise RotationError("refusing to write an empty credential")
    if shutil.which("systemd-creds") is None:
        raise RotationError("systemd-creds is not installed on this host")

    target = credentials_dir / f"{credential_name}.cred"

    # Real systemd-creds takes positional PLAINTEXT and CIPHERTEXT paths; there
    # is no --output= flag despite what the project template's shorthand
    # suggests. `-` reads the plaintext from stdin so it never touches disk.
    encrypt = ["sudo", "-n", "systemd-creds", "encrypt",
               f"--name={credential_name}", "-", str(target)]
    try:
        result = subprocess.run(
            encrypt, input=new_value.strip().encode(), capture_output=True, timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RotationError(f"systemd-creds encrypt timed out after {timeout}s") from exc
    if result.returncode != 0:
        raise RotationError(
            f"systemd-creds encrypt failed ({result.returncode}): "
            f"{result.stderr.decode(errors='replace').strip()}"
        )

    restart = ["sudo", "-n", "systemctl", "restart", unit]
    result = subprocess.run(restart, capture_output=True, timeout=timeout, check=False)
    if result.returncode != 0:
        raise RotationError(
            f"credential written but `systemctl restart {unit}` failed: "
            f"{result.stderr.decode(errors='replace').strip()}. The new value is already "
            "in place -- fix the unit and restart it, or re-run rotation with the old key "
            "from your password manager."
        )

    log.warning("credential %s rotated and %s restarted", credential_name, unit)
    return (
        f"Rotated {credential_name} and restarted {unit}. "
        "Store the new value in your password manager now -- this file is not a backup."
    )
