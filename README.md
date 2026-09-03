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

## 1. Install the dependencies

You need **git**, **uv** (which can also supply Python), and a way to identify
photos — either a local **Ollama**, or an Anthropic API key.

```bash
# uv — it manages the virtualenv and can install Python for you
curl -LsSf https://astral.sh/uv/install.sh | sh
exec $SHELL -l                       # pick up the new PATH

python3 --version                    # need 3.11 or newer
uv python install 3.12               # only if yours is older; uv will use this
```

On Debian/Ubuntu, `sudo apt install -y git curl` covers the rest. Nothing else
is needed — `uv sync` builds the virtualenv and pulls every Python dependency.

---

## 2. Choose how photos get identified

|  | **Ollama** (local) | **Claude** (hosted) |
| --- | --- | --- |
| Cost | free | ~5–15¢ per photo |
| Data | never leaves the host | photos go to the Anthropic API |
| Needs | a GPU, realistically | an API key |
| Quality | good; weaker at picking out many small cluttered objects | best |

Both are supported and you can switch later by editing one config line — your
inventory is unaffected either way. **Step 3 sets up Ollama; skip to
[step 4](#4-install-qr-organizer) if you're using Claude.**

Embeddings are always local regardless of this choice, so the "recognise a thing
I've labelled before" behaviour costs nothing and works offline in both cases.

---

## 3. Set up Ollama

**Install and start it.**

```bash
curl -fsSL https://ollama.com/install.sh | sh    # Linux; installs and starts a systemd service
brew install ollama && brew services start ollama # macOS
systemctl status ollama                           # confirm it's running
```

**Pull a vision model.** It must be a *vision* model — a text-only model will
load happily and then fail on every photo.

| Model | Roughly | Notes |
| --- | --- | --- |
| `qwen2.5vl:7b` | ~6 GB | the default in `config.toml`; the balanced choice |
| `qwen2.5vl:3b` | ~3 GB | for a smaller GPU, at some cost in accuracy |
| `qwen2.5vl:32b` | ~21 GB | if you have the VRAM for it |

```bash
ollama pull qwen2.5vl:7b
ollama list                          # confirm it's there
```

Check the current tags and sizes at [ollama.com/library](https://ollama.com/library) —
they move faster than this README. Whatever you pick, put the tag exactly as
`ollama list` prints it into `vision.ollama.model`.

**Hardware, honestly.** A 7B vision model wants roughly 8 GB of VRAM. It will
run on CPU, but expect minutes per photo rather than seconds — workable for
inventorying a few bins in the evening, tedious for a whole garage. Each photo
costs about two model calls, plus a small one for anything it's unsure about.

**Point the app at it.** In `config.toml` (see
[step 4](#4-install-qr-organizer) for where that lives):

```toml
[vision]
backend = "ollama"        # this is the line that matters; the default is "anthropic"

[vision.ollama]
base_url = "http://127.0.0.1:11434"
model = "qwen2.5vl:7b"    # exactly as `ollama list` shows it
timeout_seconds = 300     # raise it if you're on CPU and see timeouts
context_length = 8192     # raise it if replies come back unparseable
```

Ollama on a *different* machine works too: set `base_url` to that host, and set
`OLLAMA_HOST=0.0.0.0` on it so it listens beyond its own loopback.

No API key is involved anywhere in this path.

---

## 4. Install QR Organizer

**As a service (recommended).** Runs on boot, restarts on failure:

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

Your config file is now at:

| Install | Config file |
| --- | --- |
| Service | `/etc/qr-organizer/config.toml` |
| Virtualenv | `~/.config/qr-organizer/config.toml` |

Edit it now if you're using Ollama — that's the `backend = "ollama"` line from
step 3.

### Turn on visual matching

Optional but worth it: it's what lets the app recognise something it has seen
before and reuse your label instead of asking again. It pulls in torch (~2 GB).
Without it everything works, `/health` reports `degraded`, and you name more
things by hand.

```bash
uv sync --extra embeddings                                      # venv
sudo uv pip install --python /opt/qr-organizer/.venv/bin/python \
  '/opt/qr-organizer[embeddings]'                               # service
```

### If you chose Claude instead

Store the key in your password manager first — the encrypted copy is sealed to
this host and can't be recovered anywhere else.

```bash
# service
sudo install -d -m 0700 /etc/credstore.encrypted
sudo systemd-creds encrypt --name=anthropic_api_key - \
  /etc/credstore.encrypted/anthropic_api_key.cred      # paste the key, then Ctrl-D

# virtualenv
install -m 600 /dev/null ~/.config/qr-organizer/.env
printf 'ANTHROPIC_API_KEY=sk-ant-...\n' >> ~/.config/qr-organizer/.env
```

Rotating the key later: [Secrets](docs/reference.md#secrets).

---

## 5. Run it

```bash
sudo systemctl enable --now qr-organizer     # service
uv run qr-organizer                          # virtualenv
```

**Check it before you go near a bin.** `--validate-config` tests every backend
it's configured to use and tells you exactly what's wrong:

```console
$ uv run qr-organizer --validate-config
status:  ok
  [ok  ] database: ok
  [ok  ] vision: qwen2.5vl:7b on http://127.0.0.1:11434
  [ok  ] embeddings: open_clip:ViT-B-32/laion2b_s34b_b79k (512d, cpu)
  [ok  ] search: bins: 0 item(s), 0 embedded, index: sqlite-vec
  [ok  ] storage: 88.1 GiB free on /var/lib/qr-organizer
```

It exits 0 healthy, 1 degraded, 2 broken. The Ollama problems it names for you:

```
[warn] vision: http://127.0.0.1:11434 unreachable: [Errno 111] Connection refused
       → Ollama isn't running.  systemctl start ollama

[warn] vision: http://127.0.0.1:11434 is up but 'qwen2.5vl:7b' is not pulled
              (run `ollama pull qwen2.5vl:7b`)
       → the model name in config.toml doesn't match anything in `ollama list`
```

For the installed service, run it as the service account:

```bash
sudo -u qr-organizer env QR_ORGANIZER_CONFIG=/etc/qr-organizer/config.toml \
  /opt/qr-organizer/.venv/bin/qr-organizer --validate-config
```

**Getting to it in a browser.** The app binds your Tailscale address if
Tailscale is running, otherwise localhost only — never `0.0.0.0`, because it has
no login screen and its reachability *is* the security boundary.

| Situation | How you reach it |
| --- | --- |
| Tailscale on the host | `http://<tailscale-ip>:8815` — `tailscale ip -4` |
| Everything else | `ssh -L 8815:localhost:8815 <host>`, then `http://localhost:8815` |
| Want the in-app camera scanner | `sudo tailscale serve --bg 8815`, then use the `https://` address — browsers only give a page the camera over HTTPS |

You don't strictly need the in-app scanner: the QR codes contain full URLs, so
your phone's normal camera app opens the right page.

---

## 6. Start using it

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
isn't available — usually the embeddings extra, an unreachable Ollama, or a
missing API key — and it names which.

Running Ollama, specifically:

| Symptom | Cause |
| --- | --- |
| `unreachable: Connection refused` | Ollama isn't running — `systemctl start ollama` |
| `is up but '<model>' is not pulled` | the tag in `config.toml` doesn't match `ollama list` |
| Identification times out | a slow CPU run — raise `vision.ollama.timeout_seconds` |
| `reply was not JSON` in the log | context too small — raise `vision.ollama.context_length` |
| Every photo finds nothing | a text-only model — pull a *vision* model |

`journalctl -u ollama -f` shows what the model server itself is doing.

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
uv run pytest              # 126 tests, no network or API key needed
uv run ruff check src tests
```

Licensed AGPL-3.0-or-later.
