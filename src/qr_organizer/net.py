"""Bind-address selection and Tailscale detection.

`server.host = "auto"` means: bind the Tailscale interface if one is up, else
bind loopback and let the user reach it over an SSH tunnel. `0.0.0.0` is never
chosen automatically and is rejected in config validation -- this app has no
authentication, so its reachability is the security boundary.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import shutil
import socket
import subprocess

log = logging.getLogger(__name__)

#: Tailscale hands out addresses from the CGNAT range.
TAILSCALE_NET = ipaddress.ip_network("100.64.0.0/10")
LOOPBACK = "127.0.0.1"


def _from_tailscale_cli() -> str | None:
    binary = shutil.which("tailscale")
    if not binary:
        return None
    try:
        result = subprocess.run(
            [binary, "ip", "-4"], capture_output=True, timeout=5, check=False, text=True
        )
    except (subprocess.SubprocessError, OSError) as exc:
        log.debug("tailscale ip failed: %s", exc)
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        candidate = line.strip()
        if _is_tailscale_address(candidate):
            return candidate
    return None


def _from_ip_command() -> str | None:
    binary = shutil.which("ip")
    if not binary:
        return None
    try:
        result = subprocess.run(
            [binary, "-4", "-json", "addr", "show"], capture_output=True, timeout=5,
            check=False, text=True,
        )
        interfaces = json.loads(result.stdout or "[]")
    except (subprocess.SubprocessError, OSError, json.JSONDecodeError) as exc:
        log.debug("ip addr enumeration failed: %s", exc)
        return None
    for interface in interfaces:
        for address in interface.get("addr_info", []):
            local = address.get("local", "")
            if _is_tailscale_address(local):
                return local
    return None


def _is_tailscale_address(candidate: str) -> bool:
    try:
        return ipaddress.ip_address(candidate) in TAILSCALE_NET
    except ValueError:
        return False


def detect_tailscale_ip() -> str | None:
    """The host's Tailscale IPv4 address, or None if Tailscale isn't up here."""
    return _from_tailscale_cli() or _from_ip_command()


def tailscale_hostname() -> str | None:
    """The host's MagicDNS name, when `tailscale status` will tell us."""
    binary = shutil.which("tailscale")
    if not binary:
        return None
    try:
        result = subprocess.run(
            [binary, "status", "--json"], capture_output=True, timeout=5, check=False, text=True
        )
        if result.returncode != 0:
            return None
        payload = json.loads(result.stdout or "{}")
    except (subprocess.SubprocessError, OSError, json.JSONDecodeError):
        return None
    name = (payload.get("Self") or {}).get("DNSName", "")
    return name.rstrip(".") or None


def resolve_bind_host(configured: str) -> tuple[str, str]:
    """Return (host_to_bind, human explanation)."""
    if configured and configured != "auto":
        return configured, f"server.host = {configured!r} from config"
    tailscale_ip = detect_tailscale_ip()
    if tailscale_ip:
        return tailscale_ip, f"auto-detected Tailscale interface ({tailscale_ip})"
    return LOOPBACK, (
        "no Tailscale interface found; bound to localhost only. Reach it with "
        "`ssh -L <port>:localhost:<port> <host>`"
    )


def public_base_url(configured_base: str, host: str, port: int) -> str:
    """The URL to bake into printed QR codes."""
    if configured_base.strip():
        return configured_base.strip().rstrip("/")
    hostname = tailscale_hostname()
    if hostname and _is_tailscale_address(host):
        return f"http://{hostname}:{port}"
    return f"http://{host}:{port}"


def local_hostname() -> str:
    try:
        return socket.gethostname()
    except OSError:
        return "unknown"
