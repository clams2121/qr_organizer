#!/usr/bin/env bash
# One-shot installer for the systemd deployment. Idempotent; safe to re-run.
#
#   sudo ./deploy/install.sh
#
# It does NOT set the API credential — that step is manual and documented in
# the README, because the plaintext must reach your password manager first.

set -euo pipefail

APP=qr-organizer
PREFIX=/opt/$APP
SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $EUID -ne 0 ]]; then
  echo "run this with sudo" >&2
  exit 1
fi

echo "==> service account"
id -u "$APP" >/dev/null 2>&1 || useradd --system --home-dir /var/lib/$APP \
  --shell /usr/sbin/nologin "$APP"

echo "==> code at $PREFIX"
mkdir -p "$PREFIX"
cp -r "$SOURCE/src" "$SOURCE/pyproject.toml" "$SOURCE/config.default.toml" \
      "$SOURCE/README.md" "$PREFIX/"

echo "==> virtualenv"
if command -v uv >/dev/null 2>&1; then
  uv venv "$PREFIX/.venv"
  uv pip install --python "$PREFIX/.venv/bin/python" "$PREFIX"
else
  python3 -m venv "$PREFIX/.venv"
  "$PREFIX/.venv/bin/pip" install --upgrade pip
  "$PREFIX/.venv/bin/pip" install "$PREFIX"
fi

echo "==> directories"
install -d -o "$APP" -g "$APP" -m 0755 /etc/$APP /var/lib/$APP /var/log/$APP
install -d -m 0755 /var/log/service-registry   # shared with the status aggregator
chown "$APP:$APP" /var/log/service-registry || true
chown -R "$APP:$APP" "$PREFIX"

echo "==> config"
if [[ ! -f /etc/$APP/config.toml ]]; then
  install -o "$APP" -g "$APP" -m 0640 "$SOURCE/config.default.toml" /etc/$APP/config.toml
  # System-wide paths rather than the per-user XDG defaults.
  sed -i 's|^data_dir = .*|data_dir = "/var/lib/qr-organizer"|' /etc/$APP/config.toml
  echo "    wrote /etc/$APP/config.toml"
else
  echo "    /etc/$APP/config.toml already exists — left untouched"
fi

echo "==> systemd"
install -m 0644 "$SOURCE/deploy/$APP.service" /etc/systemd/system/$APP.service
install -m 0440 "$SOURCE/deploy/$APP.sudoers" /etc/sudoers.d/$APP
visudo -c -f /etc/sudoers.d/$APP
systemctl daemon-reload

cat <<'NEXT'

Installed. Remaining manual steps:

  1. Store your Anthropic API key in your password manager.
  2. Seal it for this host:
       install -d -m 0700 /etc/credstore.encrypted
       systemd-creds encrypt --name=anthropic_api_key - \
         /etc/credstore.encrypted/anthropic_api_key.cred
       (paste the key, then Ctrl-D)
  3. systemctl enable --now qr-organizer
  4. sudo -u qr-organizer /opt/qr-organizer/.venv/bin/qr-organizer --validate-config

Skip steps 1-2 entirely if you set vision.backend = "ollama".
NEXT
