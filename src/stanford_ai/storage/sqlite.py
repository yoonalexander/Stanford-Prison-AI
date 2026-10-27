import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from stanford_ai.models import World


def now() -> str:
    return datetime.now(UTC).isoformat()


def encode(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def snapshot(world: World) -> dict:
    return world.model_dump(mode="json", exclude={"history"})


def digest(value: dict) -> str:
    return hashlib.sha256(encode(value).encode("utf-8")).hexdigest()


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            status TEXT NOT NULL, reason TEXT, config TEXT NOT NULL,
            provenance TEXT NOT NULL, initial_state TEXT NOT NULL, usage TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS events (
            run_id TEXT NOT NULL REFERENCES runs(id), seq INTEGER NOT NULL,
            timestamp TEXT NOT NULL, data TEXT NOT NULL, before_hash TEXT NOT NULL,
            after_hash TEXT NOT NULL, changed_fields TEXT NOT NULL,
            PRIMARY KEY(run_id, seq)
        );
        CREATE TABLE IF NOT EXISTS snapshots (
            run_id TEXT NOT NULL REFERENCES runs(id), seq INTEGER NOT NULL,
            state TEXT NOT NULL, PRIMARY KEY(run_id, seq)
        );
        CREATE TABLE IF NOT EXISTS requests (
            run_id TEXT NOT NULL REFERENCES runs(id), ordinal INTEGER NOT NULL,
            started_at TEXT NOT NULL, finished_at TEXT, agent_id TEXT NOT NULL,
            day INTEGER NOT NULL, round INTEGER NOT NULL, prompt TEXT NOT NULL,
            reservation TEXT NOT NULL, result TEXT, PRIMARY KEY(run_id, ordinal)
        );
        CREATE TABLE IF NOT EXISTS metrics (
            run_id TEXT NOT NULL REFERENCES runs(id), seq INTEGER NOT NULL,
            data TEXT NOT NULL, PRIMARY KEY(run_id, seq)
        );
        """)

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def create(self, run_id, config, provenance, world, usage):
        state = snapshot(world)
        with self.db:
            self.db.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?)", (
                run_id, now(), now(), "running", None, encode(config), encode(provenance),
                encode(state), encode(usage)))
            self.db.execute("INSERT INTO snapshots VALUES (?,?,?)", (run_id, 0, encode(state)))

    def status(self, run_id, status, reason, usage):
        with self.db:
            self.db.execute("UPDATE runs SET status=?,reason=?,usage=?,updated_at=? WHERE id=?",
                            (status, reason, encode(usage), now(), run_id))

    def update_provenance(self, run_id, provenance):
        with self.db:
            self.db.execute("UPDATE runs SET provenance=?,updated_at=? WHERE id=?",
                            (encode(provenance), now(), run_id))

    def start_request(self, run_id, ordinal, agent_id, day, round_, prompt, reservation, usage):
        with self.db:
            self.db.execute("INSERT INTO requests VALUES (?,?,?,?,?,?,?,?,?,?)", (
                run_id, ordinal, now(), None, agent_id, day, round_, encode(prompt),
                encode(reservation), None))
            self.db.execute("UPDATE runs SET usage=?,updated_at=? WHERE id=?",
                            (encode(usage), now(), run_id))

    def finish_request(self, run_id, ordinal, result, usage):
        with self.db:
            self.db.execute("UPDATE requests SET finished_at=?,result=? WHERE run_id=? AND ordinal=?",
                            (now(), encode(result), run_id, ordinal))
            self.db.execute("UPDATE runs SET usage=?,updated_at=? WHERE id=?",
                            (encode(usage), now(), run_id))

    def record(self, run_id, event, before: dict, world: World, metrics: dict):
        after = snapshot(world)
        seq = len(world.history)
        changed = [key for key in after if before.get(key) != after[key]]
        with self.db:
            self.db.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?)", (
                run_id, seq, now(), encode(event), digest(before), digest(after), encode(changed)))
            self.db.execute("INSERT INTO snapshots VALUES (?,?,?)", (run_id, seq, encode(after)))
            self.db.execute("INSERT INTO metrics VALUES (?,?,?)", (run_id, seq, encode(metrics)))
            self.db.execute("UPDATE runs SET updated_at=? WHERE id=?", (now(), run_id))

    def runs(self) -> list[dict]:
        return [self._run(row) for row in self.db.execute("SELECT * FROM runs ORDER BY created_at DESC")]

    def run(self, run_id) -> dict:
        row = self.db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown experiment {run_id}")
        return self._run(row)

    def _run(self, row) -> dict:
        result = dict(row)
        for field in ("config", "provenance", "initial_state", "usage"):
            result[field] = json.loads(result[field])
        return result

    def events(self, run_id) -> list[dict]:
        return [dict(row) | {"data": json.loads(row["data"]),
                             "changed_fields": json.loads(row["changed_fields"])}
                for row in self.db.execute("SELECT * FROM events WHERE run_id=? ORDER BY seq",
                                           (run_id,))]

    def requests(self, run_id) -> list[dict]:
        rows = self.db.execute("SELECT * FROM requests WHERE run_id=? ORDER BY ordinal", (run_id,))
        return [dict(row) | {"prompt": json.loads(row["prompt"]),
                             "reservation": json.loads(row["reservation"]),
                             "result": json.loads(row["result"]) if row["result"] else None}
                for row in rows]

    def metrics(self, run_id) -> list[dict]:
        return [json.loads(row[0]) for row in self.db.execute(
            "SELECT data FROM metrics WHERE run_id=? ORDER BY seq", (run_id,))]

    def replay(self, run_id, seq: int | None = None) -> World:
        if seq is None:
            row = self.db.execute("SELECT seq,state FROM snapshots WHERE run_id=? ORDER BY seq DESC "
                                  "LIMIT 1", (run_id,)).fetchone()
        else:
            row = self.db.execute("SELECT seq,state FROM snapshots WHERE run_id=? AND seq=?",
                                  (run_id, seq)).fetchone()
        if row is None:
            raise KeyError("No snapshot at this event")
        state = json.loads(row["state"])
        state["history"] = [json.loads(e[0]) for e in self.db.execute(
            "SELECT data FROM events WHERE run_id=? AND seq<=? ORDER BY seq", (run_id, row["seq"]))]
        return World.model_validate(state)

    def recover(self, run_id):
        """Explicit recovery after a killed process, not automatic continuation/rebilling."""
        run = self.run(run_id)
        if run["status"] != "running":
            raise ValueError("Only abandoned running experiments need recovery")
        usage = run["usage"]
        pending = sum(r["result"] is None for r in self.requests(run_id))
        usage["unknown_usage_requests"] += pending
        self.status(run_id, "interrupted", "Recovered after process interruption; no continuation", usage)
