"""Experiment log stored in SQLite.

Each run saves its expectation BEFORE the code runs, then its outputs or its
error. Failed runs are kept. Past runs can be inspected by ID.

    python runlog.py list
    python runlog.py show 2
    python runlog.py verdict 2 "confirmed: price gap is real"
    python runlog.py export
"""

import hashlib
import json
import sqlite3
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent
DB_PATH = ROOT / "runs.sqlite"
DATA = ROOT / "data"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    status        TEXT NOT NULL,      -- started, ok, failed
    expectation   TEXT NOT NULL,      -- written before the run
    inputs        TEXT NOT NULL,      -- JSON
    outputs       TEXT,               -- JSON, or the error for a failed run
    model_version TEXT NOT NULL,
    data_version  TEXT NOT NULL,      -- hash of the data files used
    code_version  TEXT NOT NULL,      -- git commit, "+dirty" if uncommitted changes
    verdict       TEXT                -- written after reading the outputs
)
"""


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.execute(SCHEMA)
    return db


def data_version(files=None):
    """Short hash of the data files, so a run says exactly which data it saw."""
    files = files or sorted(p.name for p in DATA.glob("*.csv"))
    digest = hashlib.sha256()
    for name in files:
        digest.update(name.encode())
        digest.update((DATA / name).read_bytes())
    return digest.hexdigest()[:12]


def code_version():
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.splitlines()
        # Count new (untracked) code files too, but not the run log's own output files.
        ignore = ("runs.sqlite", "runs_export.json", "model/")
        changed = [line[3:] for line in status if not line[3:].startswith(ignore)]
        return commit + ("+dirty" if changed else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_experiment(name, expectation, inputs, model_version, fn, data_files=None):
    """Save the expectation, run fn(inputs), then save outputs or the error.

    Returns (run_id, outputs). A failure is stored and then raised again.
    """
    if not expectation or len(expectation.strip()) < 20:
        raise ValueError("Write a real expectation before running.")
    db = connect()
    cursor = db.execute(
        "INSERT INTO runs (name, started_at, status, expectation, inputs, model_version, data_version, code_version)"
        " VALUES (?, ?, 'started', ?, ?, ?, ?, ?)",
        (name, now(), expectation.strip(), json.dumps(inputs, sort_keys=True), model_version,
         data_version(data_files), code_version()),
    )
    run_id = cursor.lastrowid
    db.commit()
    print(f"Run {run_id} started: {name}")
    try:
        outputs = fn(inputs)
    except Exception:
        db.execute("UPDATE runs SET status='failed', finished_at=?, outputs=? WHERE id=?",
                   (now(), json.dumps({"error": traceback.format_exc()}), run_id))
        db.commit()
        print(f"Run {run_id} FAILED and was saved.")
        raise
    db.execute("UPDATE runs SET status='ok', finished_at=?, outputs=? WHERE id=?",
               (now(), json.dumps(outputs, sort_keys=True), run_id))
    db.commit()
    print(f"Run {run_id} saved.")
    return run_id, outputs


def get_run(run_id):
    db = connect()
    row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    db.close()
    if row is None:
        raise KeyError(f"No run with id {run_id}")
    run = dict(row)
    for key in ("inputs", "outputs"):
        run[key] = json.loads(run[key]) if run[key] else None
    return run


def list_runs():
    db = connect()
    rows = [dict(r) for r in db.execute(
        "SELECT id, name, status, model_version, data_version, started_at, verdict FROM runs ORDER BY id")]
    db.close()
    return rows


def set_verdict(run_id, text):
    db = connect()
    db.execute("UPDATE runs SET verdict=? WHERE id=?", (text, run_id))
    db.commit()
    db.close()


def export(path=ROOT / "runs_export.json"):
    runs = [get_run(r["id"]) for r in list_runs()]
    Path(path).write_text(json.dumps(runs, indent=2), encoding="utf-8")
    return path


def main(argv):
    if not argv or argv[0] == "list":
        for r in list_runs():
            print(f"{r['id']:>3}  {r['status']:7} {r['model_version']:18} data={r['data_version']}  {r['name']}"
                  + (f"\n       verdict: {r['verdict']}" if r["verdict"] else ""))
    elif argv[0] == "show":
        print(json.dumps(get_run(int(argv[1])), indent=2))
    elif argv[0] == "verdict":
        set_verdict(int(argv[1]), " ".join(argv[2:]))
    elif argv[0] == "export":
        print(f"Wrote {export()}")
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
