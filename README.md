# QR Organizer

A self-hosted inventory for the stuff in your bins. Scan a bin's QR code, take
one photo of everything laid out on the floor, and the app builds a searchable,
photo-illustrated list of what's in it — and where that bin currently lives.

The point is to stop losing track of tools and parts. Search is keyword-first:
you never need to know where something is in order to find it.

```
print labels → stick one on a bin → scan a place → scan the bin → photograph the contents → search
```

**[Full documentation →](docs/reference.md)** — how identification works, secrets
handling, every config option, health checks, design decisions.

---

## Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- An [Anthropic API key](https://console.anthropic.com/), *or* a local
  [Ollama](https://ollama.com/) install if you'd rather nothing left the host
- Linux, if you want the systemd deployment

---

## Install

**As a service (recommended).** Runs on boot, restarts on failure, keeps the API
key encrypted at rest:

```bash
git clone https://github.com/clams2121/qr_organizer.git
cd qr_organizer
sudo ./deploy/install.sh
```

**Or just in a virtualenv**, to try it out or run it by hand:

```bash
git clone https://github.com/clams2121/qr_organizer.git
cd qr_organizer
uv sync
uv run qr-organizer --setup
```

Either way, add `--extra embeddings` / install the `embeddings` extra to turn on
visual matching — it's what lets the app recognise a thing it has seen before and
reuse your label for it. It pulls in torch (~2 GB). Without it the app works
fine, reports `degraded` on `/health`, and asks you to name more things.

```bash
uv sync --extra embeddings                                      # venv
sudo uv pip install --python /opt/qr-organizer/.venv/bin/python \
  '/opt/qr-organizer[embeddings]'                               # service
```

### Set the API key

Skip this entirely if you set `vision.backend = "ollama"`.

**Service:** store the key in your password manager first — the encrypted copy is
sealed to this host and can't be recovered anywhere else. Then:

```bash
sudo install -d -m 0700 /etc/credstore.encrypted
sudo systemd-creds encrypt --name=anthropic_api_key - \
  /etc/credstore.encrypted/anthropic_api_key.cred      # paste the key, then Ctrl-D
```

**Virtualenv:**

```bash
install -m 600 /dev/null ~/.config/qr-organizer/.env
printf 'ANTHROPIC_API_KEY=sk-ant-...\n' >> ~/.config/qr-organizer/.env
```

Full details, including rotating the key from the web UI:
[Secrets](docs/reference.md#secrets).

---

## Run

```bash
sudo systemctl enable --now qr-organizer     # service
uv run qr-organizer                          # virtualenv
```

Then check it's happy — `uv run` from the checkout, or the service's own binary:

```bash
uv run qr-organizer --validate-config              # exits 0 healthy, 1 degraded, 2 broken
sudo -u qr-organizer env QR_ORGANIZER_CONFIG=/etc/qr-organizer/config.toml \
  /opt/qr-organizer/.venv/bin/qr-organizer --validate-config
```

**Getting to it in a browser.** The app binds your Tailscale address if Tailscale
is running, otherwise localhost only — never `0.0.0.0`, because it has no login
screen and its reachability *is* the security boundary.

| Situation | How you reach it |
| --- | --- |
| Tailscale on the host | `http://<tailscale-ip>:8815` — `tailscale ip -4` |
| Everything else | `ssh -L 8815:localhost:8815 <host>`, then `http://localhost:8815` |
| Want the in-app camera scanner | `sudo tailscale serve --bg 8815`, then use the `https://` address — browsers only give a page the camera over HTTPS |

You don't strictly need the in-app scanner: the QR codes contain full URLs, so
your phone's normal camera app opens the right page.

---

## Start using it

1. **Print labels.** Open `/labels`, generate a sheet, print it. Codes are
   reserved as they're printed, so no two sheets can collide. Headless
   equivalent: `uv run qr-organizer --print-sheet 24 --output labels.pdf`.
2. **Label your places.** Make a place for each storage area — "Shed — north
   wall", "Basement, under the stairs" — at `/locations`, print its placard, and
   stick it up. Places are flat; put the detail in the name.
3. **Stick a label on a bin**, then scan the place placard and the bin label, in
   that order. The bin picks up the place you just scanned. (That context
   expires after 30 minutes of inactivity, so a stale location can never get
   attached to tomorrow's scans.)
4. **Photograph the contents.** Tip the bin out, lay everything on the floor,
   take one photo. Identification takes 30 seconds or so; the page updates
   itself. You get an item per thing, each with a cropped thumbnail.
5. **Name whatever it wasn't sure about** at `/review`. Every correction you make
   teaches the matcher immediately — no retraining — so the next photo with a
   similar object reuses your label.
6. **Search for something** at `/search`. Add results to your pull list, tick
   them off as you collect them, and they're marked "in use" until you scan them
   back into any bin.

Re-photographing a bin later is the same action as step 4: new things get added,
things that have gone get flagged missing rather than deleted, and the bin's
location is left alone.

---

## Update

Your config, database, photos and credentials are never touched by an update.
The database and config schemas migrate themselves on the next start, and any
new config fields get highlighted at `/config` until you've reviewed them.

```bash
cd qr_organizer
git pull

# service — re-running the installer IS the update
sudo ./deploy/install.sh
sudo systemctl restart qr-organizer

# virtualenv
uv sync
```

Then confirm it came back up with `--validate-config`, as above.

One thing that needs a manual step: if you change `embeddings.model`, every
stored fingerprint is invalidated. The app refuses to mix models and tells you
to run `--rebuild-embeddings`.

---

## Everyday commands

Run these with `uv run` from the checkout, or as
`/opt/qr-organizer/.venv/bin/qr-organizer` for the installed service:

```bash
qr-organizer                                   # start the server
qr-organizer --validate-config                 # check config and every backend
qr-organizer --setup                           # create config, directories, database
qr-organizer --print-sheet 24 --output x.pdf   # a sheet of labels without a browser
qr-organizer --rebuild-embeddings              # after changing the embedding model
```

## Pages

| | |
| --- | --- |
| `/` | search box, what needs your attention, recent activity |
| `/scan` | in-app camera scanner |
| `/review` | things to name, and returns to confirm |
| `/labels` | print sheets of QR codes |
| `/status` | health checks, recent errors, recent scans |
| `/config` | edit configuration, rotate the API key |
| `/health` | JSON readiness, for monitoring |

## If something looks wrong

Start at `/status` — health checks, the recent log, and the last identification
runs are all on one page. The full log is at `/var/log/qr-organizer/app.log`
(or `<data_dir>/logs/app.log` if that isn't writable; the app says which at
startup). `journalctl -u qr-organizer -f` for the service.

`degraded` on `/health` is normal and specific: it means something it needs
isn't available — usually a missing API key or the embeddings extra — and it
names which.

---

## Where things live

| | Service | Virtualenv |
| --- | --- | --- |
| Config | `/etc/qr-organizer/config.toml` | `~/.config/qr-organizer/config.toml` |
| Database, photos | `/var/lib/qr-organizer/` | `~/.local/share/qr-organizer/` |
| Log | `/var/log/qr-organizer/app.log` | same, else `<data_dir>/logs/` |

## Development

```bash
uv sync --extra dev
uv run pytest              # 110 tests, no network or API key needed
uv run ruff check src tests
```

Licensed AGPL-3.0-or-later.
