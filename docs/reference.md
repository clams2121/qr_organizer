# QR Organizer — reference

Full documentation. If you just want to install it and start scanning, the
[README](../README.md) is the short version.

A self-hosted inventory for the stuff in your bins. Scan a bin's QR code, take
one photo of everything laid out on the floor, and the app builds a searchable,
photo-illustrated list of what's in it — and where that bin currently lives.

The point is to stop losing track of tools and parts. Search is keyword-first:
you should never need to know where something is in order to find it.

```
scan a place code  →  scan a bin code  →  photograph the contents  →  search for anything
```

---

## Contents

- [What it does](#what-it-does)
- [How identification works](#how-identification-works)
- [Returns](#returns)
- [Install](#install)
  - [systemd (recommended)](#systemd-recommended)
  - [Plain venv](#plain-venv)
- [Secrets](#secrets)
  - [systemd-creds walkthrough](#systemd-creds-walkthrough)
  - [.env, for venv runs](#env-for-venv-runs)
  - [Rotating the key from the web form](#rotating-the-key-from-the-web-form)
- [Reaching the config web form](#reaching-the-config-web-form)
- [Labels and QR codes](#labels-and-qr-codes)
- [Configuration](#configuration)
- [Health, status and the service registry](#health-status-and-the-service-registry)
- [CLI](#cli)
- [Design decisions](#design-decisions)
- [Development](#development)

---

## What it does

**Bins.** A physical tote with a QR code, a human-readable code printed next to
it, a current location, and a list of items.

**Places.** Flat, no hierarchy: "Shed — north wall", "neighbour's garage".
Specificity comes from the name. Scanning a place code makes it the *active
location*; every bin you claim afterwards is tagged with it, until you scan a
different place or the context times out (30 minutes of inactivity by default).

**Items.** One distinct thing found in a bin photo — a label, a thumbnail
cropped from the photo, a visual fingerprint, and a status.

**Loans.** A virtual bin belonging to a person rather than a place. Loaned items
show *who has them* in search results, and come back by being scanned into any
bin.

**Pull list.** A shopping-cart-style checklist. Add things from search results,
then tick them off as you physically pick them up. Ticking marks the item *in
use* until it is scanned back into a bin.

Three rules the app holds to strictly, all of them the same rule really — the
app records what it saw, and you decide what it means:

- **A re-inventory scan never moves a bin.** Photographing a bin's contents
  updates its contents and nothing else. Location changes are always a
  deliberate action.
- **Items are never auto-deleted.** Something not detected in the latest photo is
  flagged *missing*, because missing usually means moved, loaned or misplaced —
  not gone from the world.
- **A returned item is confirmed, never assumed.** If something you had out on
  loan, in use, or flagged missing turns up in a bin photo, the app queues the
  question rather than deciding. Its status doesn't budge until you say "it's
  back" — see [Returns](#returns).

---

## How identification works

One photo goes through three passes, plus a visual-similarity lookup between
them.

| Pass | Calls | What it does |
| --- | --- | --- |
| **1. Enumerate** | 1 whole-image | Lists the distinct items in the layout. A whole-image pass is what gets the *count* right — it can see that three sockets in a bag are one item. |
| **2. Locate** | 1 whole-image | Draws a bounding box around each enumerated name. Splitting naming from boxing matters: models box a thing you've already named far better than they name and box in one breath, and every item needs a real box because the crop becomes the thumbnail and the fingerprint. |
| **RAG lookup** | none | Every crop is embedded and compared against every thumbnail you've already labelled. A close match reuses that exact label. |
| **3. Verify** | 1 small call per *uncertain* crop | A focused look at one crop, offered the near-miss labels as candidates. Most crops never reach it. |

Roughly 2 whole-image calls plus a handful of small ones — on a busy tote photo
with Claude, about 5–15 US cents. Raise `vision.enumerate_passes` to union
repeated independent enumerations for better recall on very cluttered layouts,
at one extra whole-image call each.

**The label library updates live.** Correcting a label changes what future
photos will suggest, immediately — no retraining, no batch job. That is what
makes generic catch-all labels work: name one pile "robot kit parts" and
visually similar parts get that label automatically from then on.

**The accuracy bar is deliberately relaxed.** "wrench" is a good label. The bar
is "good enough to pick the right thing out of a grid of thumbnails". Anything
the pipeline can't name confidently is queued at `/review` for you to name,
rather than guessed at or silently dropped.

**Two backends**, chosen with `vision.backend`:

- `anthropic` — Claude vision. Best identification quality; needs an API key
  and outbound network.
- `ollama` — a local vision model (default `qwen2.5vl:7b`). No API key, no
  per-photo cost, no data leaves the host; noticeably weaker at enumerating many
  small cluttered objects, and wants a GPU.

Embeddings are always local ([open-clip](https://github.com/mlfoundations/open_clip)),
so the visual-matching library costs nothing per lookup and works offline in
both cases.

---

## Returns

A visual match is evidence, not a decision. When a re-inventory photo turns up
something you had put elsewhere, the app says so and waits:

> **Did these come back? (2)**
> These turned up in a photo of this bin, but you had them somewhere else.
> Nothing has changed yet.
>
> `wrench` — with Dave next door → **It's back** · Still out
> `roll of tape` — in use → **It's back** · Still out
>
> **All 2 are back** · None of them are

Until you answer, the item keeps exactly the status you gave it: still on loan
to Dave, still in use, still flagged missing. Confirming checks it into that bin
and takes it off Dave's loan (the loan itself stays open until everything on it
is back); dismissing leaves everything untouched and records that you checked.
Both outcomes land in the item's history, so a wrong match costs one tap and
nothing else.

The details:

- **The same question is never asked twice.** Photographing a bin three times
  queues one question per item, not three.
- **Checking an item in by hand clears the question**, because it's moot.
- **A queued item can't also be flagged missing** on the next scan — no
  contradictory flags.
- Pending returns appear on the bin page, on the item's own page, and in
  `/review` alongside the items waiting for a label. One button confirms or
  dismisses a whole bin's worth at once.

This applies to all three states — *loaned*, *in use*, and *missing*. A missing
item reappearing is arguably the app correcting its own earlier guess, and you
could reasonably let that one auto-resolve; it's a one-line change in
`_reconcile` if you'd rather. The bulk-confirm button exists so that the
common case (you re-photograph a shelf and everything's there) is a single tap.

---

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

### systemd (recommended)

```bash
git clone https://github.com/clams2121/qr_organizer.git
cd qr_organizer
sudo ./deploy/install.sh
```

That creates a `qr-organizer` service account, installs the code to
`/opt/qr-organizer`, builds a venv, writes `/etc/qr-organizer/config.toml`
(without overwriting an existing one), and installs the unit and sudoers files.
It deliberately does **not** set the API credential — see
[Secrets](#secrets) — because the plaintext needs to reach your password
manager first.

Then:

```bash
sudo systemctl enable --now qr-organizer
sudo -u qr-organizer /opt/qr-organizer/.venv/bin/qr-organizer --validate-config
```

To enable visual matching (a ~2 GB torch install, worth it):

```bash
sudo uv pip install --python /opt/qr-organizer/.venv/bin/python \
  '/opt/qr-organizer[embeddings]'
sudo systemctl restart qr-organizer
```

(`uv venv` does not put `pip` inside the virtualenv, so install through `uv`.)

The unit uses `Type=notify` with `WatchdogSec=30`. The app pings the watchdog
only while its critical path still works, so a wedged process gets restarted
rather than kept alive by a dumb heartbeat. Restarts are capped at 5 per 5
minutes, after which systemd gives up rather than looping forever.

> **One deviation from the standard unit block:** `StartLimitIntervalSec` and
> `StartLimitBurst` are in `[Unit]`, not `[Service]`. Modern systemd only reads
> them there; in `[Service]` they parse with a deprecation warning and are
> ignored.

### Plain venv

```bash
git clone https://github.com/clams2121/qr_organizer.git
cd qr_organizer
uv sync                      # add --extra embeddings for visual matching
uv run qr-organizer --setup
uv run qr-organizer
```

Config lands in `~/.config/qr-organizer/config.toml`, data in
`~/.local/share/qr-organizer/`. Logs go to `/var/log/qr-organizer/app.log` when
that's writable and fall back to `<data_dir>/logs/app.log` when it isn't — the
app says which, loudly, at startup.

---

## Secrets

Only the `anthropic` backend needs a secret. With `vision.backend = "ollama"`
you can skip this whole section.

The app looks in three places, in order:

1. `$CREDENTIALS_DIRECTORY/anthropic_api_key` — systemd-creds. The production
   path.
2. `$ANTHROPIC_API_KEY`.
3. `~/.config/qr-organizer/.env`, which **must be mode 600**. A looser mode is
   refused and logged rather than quietly used.

> **Back the plaintext up in your password manager.** Neither the `.cred` file
> nor the `.env` file is a backup. A systemd-creds blob is sealed to this host's
> key/TPM and cannot be decrypted anywhere else — if the host dies, so does that
> copy of the key.

### systemd-creds walkthrough

```bash
# 1. Put the key in your password manager first. Really.

# 2. Create the credential store (root-only, once per host).
sudo install -d -m 0700 /etc/credstore.encrypted

# 3. Seal the key. `-` reads plaintext from stdin so it never touches disk;
#    paste the key, then press Ctrl-D.
sudo systemd-creds encrypt --name=anthropic_api_key - \
  /etc/credstore.encrypted/anthropic_api_key.cred

# 4. Confirm it decrypts on this host.
sudo systemd-creds decrypt --name=anthropic_api_key \
  /etc/credstore.encrypted/anthropic_api_key.cred - | head -c 8; echo '...'

# 5. Restart and check.
sudo systemctl restart qr-organizer
curl -s localhost:8815/health | jq .checks.vision
```

The unit already carries the matching line:

```ini
LoadCredentialEncrypted=anthropic_api_key:/etc/credstore.encrypted/anthropic_api_key.cred
```

systemd decrypts it at start into a private tmpfs visible only to this unit, and
exports `$CREDENTIALS_DIRECTORY`. Nothing writes the plaintext to disk, and no
desktop session or keyring is involved.

### .env, for venv runs

```bash
install -m 600 /dev/null ~/.config/qr-organizer/.env
printf 'ANTHROPIC_API_KEY=sk-ant-...\n' >> ~/.config/qr-organizer/.env
```

`.env` is gitignored. It is still not a backup.

### Rotating the key from the web form

`/config` has a rotation form. The web process itself stays unprivileged; two
narrowly-scoped sudo rules (`../deploy/qr-organizer.sudoers`) grant it exactly two
command lines and nothing else:

```
/usr/bin/systemd-creds encrypt --name=anthropic_api_key - /etc/credstore.encrypted/anthropic_api_key.cred
/usr/bin/systemctl restart qr-organizer.service
```

Both name every argument, so the service account can't encrypt to a different
path, name a different credential, or restart a different unit. Rotation
re-encrypts in place and restarts the unit, so the previous `.cred` is gone the
moment the new one lands — which is exactly why step 1 of the walkthrough is
"put it in your password manager".

> The project template writes this command as `systemd-creds encrypt
> --name=<X> --output=<path>`. Real `systemd-creds` has no `--output` flag; it
> takes positional `PLAINTEXT CIPHERTEXT` paths. The sudoers file uses the real
> syntax.

Two knobs if you'd rather not have this at all: set
`secrets.allow_web_rotation = false`, and add `NoNewPrivileges=yes` to the unit
(it is deliberately absent, because it would block the sudo call).

*Not built, noted for later:* phone-based split-key / session unlock —
Vault-style Shamir sharing, or Tang/Clevis presence unlock.

---

## Reaching the config web form

`server.host = "auto"` means: **bind the Tailscale interface if one is up,
otherwise bind loopback.** It never binds `0.0.0.0` — that value is rejected
outright in config validation, and the app has no authentication, so its
reachability *is* the security boundary.

**Over Tailscale.** Check what it picked:

```bash
curl -s localhost:8815/health >/dev/null && systemctl status qr-organizer | grep serving
tailscale ip -4
```

Then open `http://<tailscale-ip>:8815/config`, or use the MagicDNS name shown on
`/status`.

**Over SSH, when Tailscale isn't running here:**

```bash
ssh -L 8815:localhost:8815 <host>
# then open http://localhost:8815/config
```

**The in-app camera scanner needs HTTPS.** Browsers only expose the camera on a
secure origin. Over plain HTTP the `/scan` page detects this and says so rather
than showing a dead viewfinder. To fix it:

```bash
sudo tailscale serve --bg 8815
```

and use the `https://…ts.net` address it prints. Until then, point your phone's
own camera app at a label — the QR codes contain a full URL, so they open the
right page directly with no app involved.

---

## Labels and QR codes

Codes are minted in **batches and printed ahead of time**, not generated per
bin. Grab the next label off the sheet when you fill a tote, scan it, and the
bin comes into being.

- `/labels` generates a printable PDF sheet and reserves those codes so no two
  sheets can ever collide.
- Every label carries the QR code **and** the human-readable code (`BIN-0042`)
  beside it, so a bin can be identified across the room without a phone.
- Places get one big placard each: `/labels/single/LOC-0001.pdf`.
- Headless: `qr-organizer --print-sheet 24 --output labels.pdf`.

The QR payload is the full URL `http://<host>:<port>/s/BIN-0042`. The in-app
scanner strips it back to a code, and a phone's stock camera app opens it
directly.

> If the server's address ever changes, set `server.base_url` **before** printing
> more sheets. Already-printed labels keep the old address baked in.

---

## Configuration

TOML, at `~/.config/qr-organizer/config.toml` (or `/etc/qr-organizer/config.toml`
under systemd, via `QR_ORGANIZER_CONFIG`). On first run it's copied from
`config.default.toml`; **your copy is never overwritten**.

When a new release adds fields, they're merged in with defaults, written back,
logged as a warning, and highlighted in the config web page until you press
"I've reviewed these". Until then, `/health` reports `degraded` — a silent
migration is exactly the kind of thing that bites you six months later.

The settings worth knowing about:

| Key | Default | Notes |
| --- | --- | --- |
| `server.host` | `"auto"` | Tailscale if present, else `127.0.0.1`. `0.0.0.0` is refused. |
| `server.port` | `8815` | |
| `server.base_url` | `""` | Baked into printed QR codes. Derived from the bind address when empty. |
| `scanning.location_context_timeout_minutes` | `30` | After this much inactivity, a fresh place scan is required before new bins get tagged. |
| `search.include_in_use_by_default` | `true` | In-use items shown with a badge rather than hidden. |
| `vision.backend` | `"anthropic"` | or `"ollama"` |
| `vision.anthropic.model` | `"claude-opus-5"` | `claude-sonnet-5` is cheaper if you'd rather trade some recall for cost. |
| `vision.anthropic.effort` | `"high"` | `low`…`max`. Directly controls how much thinking each pass does. |
| `vision.enumerate_passes` | `1` | Raise to 2–3 to union independent enumerations of a very cluttered photo. |
| `embeddings.backend` | `"clip"` | `"none"` disables visual matching entirely. |
| `embeddings.match_threshold` | `0.86` | Cosine similarity above which a past label is auto-suggested. |
| `labels.sheet_columns` / `sheet_rows` | `3` / `8` | 24 labels per page. |
| `registry.dir` | `/var/log/service-registry` | Shared with the status aggregator. |

Editing config from `/config` validates before writing, so a bad edit can't take
the service down on the next start. Changes to the vision backend, the embedding
model or the binding need a restart.

**Changing `embeddings.model` invalidates every stored fingerprint.** The app
refuses to mix dimensions and tells you to run `qr-organizer
--rebuild-embeddings`.

---

## Health, status and the service registry

`GET /health` is **readiness**-focused: it answers "can this service do its job
right now?", not "is the process alive".

```json
{
  "status": "degraded",
  "checks": {
    "database":         {"status": "ok",       "detail": "ok"},
    "vision":           {"status": "degraded", "detail": "no API key resolved"},
    "embeddings":       {"status": "ok",       "detail": "open_clip:ViT-B-32/… (512d, cpu)"},
    "search":           {"status": "ok",       "detail": "bins: 412 item(s), 412 embedded, index: sqlite-vec"},
    "storage":          {"status": "ok",       "detail": "88.1 GiB free on /var/lib/qr-organizer"},
    "service_registry": {"status": "ok",       "detail": "last written 2026-08-22T09:14:03+00:00"},
    "config":           {"status": "ok",       "detail": "loaded from /etc/qr-organizer/config.toml"}
  },
  "last_success": "2026-08-22T09:02:11+00:00"
}
```

`degraded` returns HTTP 200 (the service is still worth talking to); only a hard
`error` returns 503.

`/status` is the same information for humans, alongside the recent log, the last
identification runs, the scan history, and where the service is bound — in the
same app and the same nav as `/config`, so a config mistake and the error it
caused are one click apart.

**Service registry.** If `/var/log/service-registry/` exists, the app writes
`qr-organizer.json` there at startup and every 7 minutes:

```json
{"name": "qr-organizer", "host": "100.x.y.z", "port": 8815,
 "health_url": "…/health", "log_path": "/var/log/qr-organizer/app.log",
 "config_url": "…/config", "status_url": "…/status", "status": "running"}
```

The periodic refresh is what makes a *later*-installed status aggregator pick up
an already-running service without a restart. If the directory doesn't exist,
nothing is written and nothing complains. If the app is too broken to start at
all, it still tries to drop a `{"status": "error"}` breadcrumb there before
giving up.

**Notifications** are deliberately not wired up per-service. The status
aggregator is the single place that watches everything and fires alerts.

---

## CLI

```
qr-organizer                          start the server
qr-organizer --setup                  create config, directories and database, then exit
qr-organizer --validate-config        check config and every backend; non-zero on a problem
qr-organizer --rebuild-embeddings     recompute every fingerprint (after changing the model)
qr-organizer --print-sheet 24 -o x.pdf   reserve 24 codes and write a printable sheet
qr-organizer --host 127.0.0.1 --port 9000   one-off binding override
```

`--validate-config` exits `0` when healthy, `1` when degraded, `2` when a
critical check fails — usable straight from a deploy script.

---

## Design decisions

Where the spec offered a choice, here's what was picked and why.

**Location timeout: 30 minutes.** Long enough to work through a shelf of bins
uninterrupted, short enough that yesterday's context never silently tags today's
scans. On expiry the context is *deleted* on read, so a stale location can't be
picked up later.

**Returns are confirmed, not inferred.** The first cut of this auto-returned any
matched item that had been loaned, in use or missing — the reasoning being that
seeing it in the bin *is* it being scanned back in. That was the wrong default:
it lets one bad embedding match silently tell you Dave gave your socket set back
when he didn't, and you'd never know to look. The app now queues the question
and changes nothing until answered. The cost is a tap; the bulk action makes it
one tap per bin.

**In-use items stay in search results with a badge.** Knowing that the socket set
exists but Dave has it is almost always the answer you wanted; hiding it makes
the app look like the item vanished. A filter narrows to in-bin-only when you
specifically want a list of things you can go and fetch.

**Search is a source registry, not a bin query.** Everything above the storage
layer talks to `SearchSource` — keyword search, vector search, thumbnails,
health — with the bin inventory registered as one implementation, and results
carrying a source id. A future item database (the sidelined whole-home
inventory) becomes one class and one `register()` call. There is no plugin
loader, no entry points, and no per-source config machinery, because that would
be real machinery built for a consumer that doesn't exist yet.

**SQLite with `sqlite-vec`.** One file to back up, no daemon. `item_embeddings`
holds the vectors as BLOBs and is the source of truth; the `vec0` index is
maintained alongside it when the extension loads, and an exact NumPy scan over
the same BLOBs runs when it doesn't. A test asserts the two paths agree.

**Device identity without authentication.** A cookie identifies the browser (so
the active location and the pull list belong to the phone in your hand, not the
whole household), and `tailscale whois` on the source IP gives an informal
name for the scan log. It is attribution, not access control — anyone who can
reach the port can use the app, which is the whole reason it binds to Tailscale
or loopback.

**Fail loud.** No config at all is a hard exit (78, `EX_CONFIG`). A degraded
dependency keeps the app running, reports `degraded`, and shows up on the status
page. A model reply that doesn't match its schema raises after one retry rather
than being coerced into something plausible. A crop the model won't name goes to
`/review` instead of getting a guessed label.

---

## Development

```bash
uv sync --extra dev
uv run pytest                    # 110 tests, no network, no API key needed
uv run ruff check src tests
```

The vision backend and the embedder are both faked in the test suite, so the
whole pipeline — enumerate, locate, crop, embed, RAG lookup, reconcile — runs
offline and deterministically against synthetic photos. The fakes implement the
same protocols the real backends do, so a change that breaks the contract breaks
the tests.

Layout:

```
src/qr_organizer/
├── cli.py, runtime.py         entry point and application wiring
├── config.py                  TOML load, schema migration, validation
├── db.py                      schema, connections, vector index
├── pipeline.py                enumerate → locate → verify + RAG
├── labels.py                  QR sheet PDFs
├── health.py, watchdog.py     readiness checks, sd_notify
├── registry.py, net.py        service registry, Tailscale detection
├── secrets.py                 credential resolution and rotation
├── vision/                    Claude and Ollama backends behind one contract
├── embeddings/                local open-clip
├── search/                    SearchSource protocol + the bin implementation
├── services/                  bins, items, locations, loans, pull list, scans
└── web/                       Flask app, views, templates, static
```

Third-party bundled asset: [jsQR](https://github.com/cozmo/jsQR) 1.4.0
(Apache-2.0), in `web/static/vendor/`, used as the QR decoding fallback where
the browser has no native `BarcodeDetector`.
