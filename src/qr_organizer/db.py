"""SQLite storage: schema, migrations, connections, and vector-index plumbing.

The database is a single file so a backup is `cp inventory.db`. Vectors are
stored as raw BLOBs in `item_embeddings` -- that table is the source of truth.
When the `sqlite-vec` extension loads we additionally maintain a `vec0` index
over the same vectors for fast nearest-neighbour search; when it doesn't, the
search layer falls back to a NumPy scan over the same BLOBs. Either way there
is exactly one copy of the data that matters.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .errors import StorageError

log = logging.getLogger(__name__)

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS locations (
    id          INTEGER PRIMARY KEY,
    code        TEXT NOT NULL UNIQUE,
    name        TEXT NOT NULL,
    notes       TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS bins (
    id               INTEGER PRIMARY KEY,
    code             TEXT NOT NULL UNIQUE,
    label            TEXT NOT NULL DEFAULT '',
    location_id      INTEGER REFERENCES locations(id) ON DELETE SET NULL,
    location_set_at  TEXT,
    notes            TEXT NOT NULL DEFAULT '',
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bins_location ON bins(location_id);

CREATE TABLE IF NOT EXISTS loans (
    id           INTEGER PRIMARY KEY,
    code         TEXT NOT NULL UNIQUE,
    person_name  TEXT NOT NULL,
    notes        TEXT NOT NULL DEFAULT '',
    status       TEXT NOT NULL DEFAULT 'open',
    photo_id     INTEGER,
    created_at   TEXT NOT NULL,
    closed_at    TEXT
);

CREATE TABLE IF NOT EXISTS photos (
    id          INTEGER PRIMARY KEY,
    path        TEXT NOT NULL,
    kind        TEXT NOT NULL,
    bin_id      INTEGER REFERENCES bins(id) ON DELETE SET NULL,
    loan_id     INTEGER REFERENCES loans(id) ON DELETE SET NULL,
    session_id  INTEGER,
    width       INTEGER NOT NULL DEFAULT 0,
    height      INTEGER NOT NULL DEFAULT 0,
    sha256      TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_photos_bin ON photos(bin_id);

CREATE TABLE IF NOT EXISTS items (
    id                INTEGER PRIMARY KEY,
    label             TEXT NOT NULL,
    description       TEXT NOT NULL DEFAULT '',
    status            TEXT NOT NULL DEFAULT 'in_bin',
    bin_id            INTEGER REFERENCES bins(id) ON DELETE SET NULL,
    loan_id           INTEGER REFERENCES loans(id) ON DELETE SET NULL,
    home_bin_id       INTEGER REFERENCES bins(id) ON DELETE SET NULL,
    thumbnail_path    TEXT NOT NULL DEFAULT '',
    source_photo_id   INTEGER REFERENCES photos(id) ON DELETE SET NULL,
    bbox_json         TEXT NOT NULL DEFAULT '',
    label_source      TEXT NOT NULL DEFAULT 'ai',
    label_confidence  REAL NOT NULL DEFAULT 0.0,
    needs_review      INTEGER NOT NULL DEFAULT 0,
    notes             TEXT NOT NULL DEFAULT '',
    quantity          INTEGER NOT NULL DEFAULT 1,
    first_seen_at     TEXT NOT NULL,
    last_seen_at      TEXT NOT NULL,
    status_changed_at TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_items_bin ON items(bin_id);
CREATE INDEX IF NOT EXISTS idx_items_loan ON items(loan_id);
CREATE INDEX IF NOT EXISTS idx_items_status ON items(status);
CREATE INDEX IF NOT EXISTS idx_items_review ON items(needs_review);

CREATE VIRTUAL TABLE IF NOT EXISTS items_fts USING fts5(
    label,
    description,
    notes,
    content='items',
    content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS items_fts_insert AFTER INSERT ON items BEGIN
    INSERT INTO items_fts(rowid, label, description, notes)
    VALUES (new.id, new.label, new.description, new.notes);
END;
CREATE TRIGGER IF NOT EXISTS items_fts_delete AFTER DELETE ON items BEGIN
    INSERT INTO items_fts(items_fts, rowid, label, description, notes)
    VALUES ('delete', old.id, old.label, old.description, old.notes);
END;
CREATE TRIGGER IF NOT EXISTS items_fts_update AFTER UPDATE ON items BEGIN
    INSERT INTO items_fts(items_fts, rowid, label, description, notes)
    VALUES ('delete', old.id, old.label, old.description, old.notes);
    INSERT INTO items_fts(rowid, label, description, notes)
    VALUES (new.id, new.label, new.description, new.notes);
END;

CREATE TABLE IF NOT EXISTS item_embeddings (
    item_id     INTEGER PRIMARY KEY REFERENCES items(id) ON DELETE CASCADE,
    model       TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    vector      BLOB NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS item_events (
    id          INTEGER PRIMARY KEY,
    item_id     INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    event       TEXT NOT NULL,
    detail      TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_item_events_item ON item_events(item_id);

CREATE TABLE IF NOT EXISTS scan_events (
    id           INTEGER PRIMARY KEY,
    kind         TEXT NOT NULL,
    code         TEXT NOT NULL DEFAULT '',
    bin_id       INTEGER,
    location_id  INTEGER,
    item_id      INTEGER,
    device_key   TEXT NOT NULL DEFAULT '',
    device_ip    TEXT NOT NULL DEFAULT '',
    device_name  TEXT NOT NULL DEFAULT '',
    user_agent   TEXT NOT NULL DEFAULT '',
    detail       TEXT NOT NULL DEFAULT '',
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_scan_events_created ON scan_events(created_at);

CREATE TABLE IF NOT EXISTS location_context (
    device_key   TEXT PRIMARY KEY,
    location_id  INTEGER NOT NULL REFERENCES locations(id) ON DELETE CASCADE,
    set_at       TEXT NOT NULL,
    last_used_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pull_list (
    id              INTEGER PRIMARY KEY,
    session_key     TEXT NOT NULL,
    item_id         INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    added_at        TEXT NOT NULL,
    checked_off_at  TEXT,
    UNIQUE(session_key, item_id)
);
CREATE INDEX IF NOT EXISTS idx_pull_list_session ON pull_list(session_key);

CREATE TABLE IF NOT EXISTS label_stock (
    id          INTEGER PRIMARY KEY,
    code        TEXT NOT NULL UNIQUE,
    kind        TEXT NOT NULL,
    sheet_id    TEXT NOT NULL DEFAULT '',
    printed_at  TEXT,
    assigned_at TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_label_stock_kind ON label_stock(kind, assigned_at);

CREATE TABLE IF NOT EXISTS pending_returns (
    id             INTEGER PRIMARY KEY,
    item_id        INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    bin_id         INTEGER NOT NULL REFERENCES bins(id) ON DELETE CASCADE,
    session_id     INTEGER,
    prior_status   TEXT NOT NULL,
    confidence     REAL NOT NULL DEFAULT 0.0,
    resolution     TEXT,
    resolved_at    TEXT,
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pending_returns_open
    ON pending_returns(item_id, resolved_at);
CREATE INDEX IF NOT EXISTS idx_pending_returns_bin
    ON pending_returns(bin_id, resolved_at);

CREATE TABLE IF NOT EXISTS inventory_sessions (
    id              INTEGER PRIMARY KEY,
    bin_id          INTEGER REFERENCES bins(id) ON DELETE CASCADE,
    kind            TEXT NOT NULL DEFAULT 'bin',
    status          TEXT NOT NULL DEFAULT 'pending',
    detected_count  INTEGER NOT NULL DEFAULT 0,
    added_count     INTEGER NOT NULL DEFAULT 0,
    matched_count   INTEGER NOT NULL DEFAULT 0,
    missing_count   INTEGER NOT NULL DEFAULT 0,
    returns_count   INTEGER NOT NULL DEFAULT 0,
    error           TEXT NOT NULL DEFAULT '',
    device_key      TEXT NOT NULL DEFAULT '',
    started_at      TEXT NOT NULL,
    finished_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_bin ON inventory_sessions(bin_id);
"""


class Database:
    """Thread-local SQLite connections with the vec extension loaded when present."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._local = threading.local()
        self._write_lock = threading.Lock()
        self.vec_available = False
        self.vec_error: str | None = None
        self.vec_dim: int | None = None

    # -- lifecycle --------------------------------------------------------
    def initialise(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = self.connection()
        with self._write_lock:
            conn.executescript(SCHEMA)
            current = self.get_meta("schema_version")
            if current is None:
                self.set_meta("schema_version", str(SCHEMA_VERSION))
            elif int(current) > SCHEMA_VERSION:
                raise StorageError(
                    f"database at {self.path} was written by a newer version "
                    f"(schema {current} > {SCHEMA_VERSION}); refusing to downgrade"
                )
            elif int(current) < SCHEMA_VERSION:
                # The schema script above is idempotent, so it has already
                # created whatever the newer version added. Record the bump and
                # add any columns that a CREATE TABLE IF NOT EXISTS cannot.
                self._add_missing_columns(conn)
                self.set_meta("schema_version", str(SCHEMA_VERSION))
                log.warning(
                    "database schema upgraded from %s to %d", current, SCHEMA_VERSION
                )
            conn.commit()
        recorded_dim = self.get_meta("embedding_dim")
        self.vec_dim = int(recorded_dim) if recorded_dim else None
        if self.vec_available and self.vec_dim:
            self._ensure_vec_table(conn, self.vec_dim)
        log.info("database ready at %s (vector index: %s)", self.path,
                 "sqlite-vec" if self.vec_available else f"numpy fallback ({self.vec_error})")

    def _add_missing_columns(self, conn: sqlite3.Connection) -> None:
        """Bring an older table up to date. `ALTER TABLE ADD COLUMN` only."""
        additions = {"inventory_sessions": {"returns_count": "INTEGER NOT NULL DEFAULT 0"}}
        for table, columns in additions.items():
            present = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
            for name, definition in columns.items():
                if name not in present:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")
                    log.info("added column %s.%s", table, name)

    def connection(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            return conn
        conn = sqlite3.connect(self.path, timeout=30.0, isolation_level=None,
                               check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        self._load_vec(conn)
        self._local.conn = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def _load_vec(self, conn: sqlite3.Connection) -> None:
        try:
            import sqlite_vec

            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            self.vec_available = True
        except Exception as exc:  # extension genuinely optional; fallback is exact
            self.vec_available = False
            self.vec_error = f"{type(exc).__name__}: {exc}"

    # -- vector index -----------------------------------------------------
    def ensure_vector_index(self, dim: int) -> None:
        """Bind the database to an embedding dimension, creating the vec0 index."""
        recorded = self.get_meta("embedding_dim")
        if recorded and int(recorded) != dim:
            raise StorageError(
                f"stored embeddings are {recorded}-dimensional but the configured model "
                f"produces {dim}; run `qr-organizer --rebuild-embeddings` after changing "
                "embeddings.model"
            )
        if not recorded:
            self.set_meta("embedding_dim", str(dim))
        self.vec_dim = dim
        if self.vec_available:
            self._ensure_vec_table(self.connection(), dim)

    def _ensure_vec_table(self, conn: sqlite3.Connection, dim: int) -> None:
        try:
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS item_vectors USING vec0("
                f"item_id INTEGER PRIMARY KEY, embedding float[{dim}])"
            )
        except sqlite3.Error as exc:
            self.vec_available = False
            self.vec_error = f"vec0 table unavailable: {exc}"
            log.warning("sqlite-vec loaded but the index could not be created: %s", exc)

    # -- helpers ----------------------------------------------------------
    def get_meta(self, key: str) -> str | None:
        row = self.connection().execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.connection().execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    @contextmanager
    def write(self) -> Iterator[sqlite3.Connection]:
        """Serialised write transaction. Rolls back and re-raises on failure."""
        conn = self.connection()
        with self._write_lock:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except Exception:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")

    def query(self, sql: str, params: tuple | dict = ()) -> list[sqlite3.Row]:
        return list(self.connection().execute(sql, params).fetchall())

    def query_one(self, sql: str, params: tuple | dict = ()) -> sqlite3.Row | None:
        return self.connection().execute(sql, params).fetchone()

    def health_check(self) -> tuple[bool, str]:
        try:
            self.connection().execute("SELECT 1 FROM items LIMIT 1").fetchone()
            return True, "ok"
        except sqlite3.Error as exc:
            return False, f"{type(exc).__name__}: {exc}"
