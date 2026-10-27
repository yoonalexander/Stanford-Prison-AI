import json
from pathlib import Path

import pandas as pd

from stanford_ai.metrics.compute import action_frame


def export_run(store, run_id, directory: str | Path) -> Path:
    out = Path(directory) / run_id
    out.mkdir(parents=True, exist_ok=True)
    # Raw and derived outputs have separate filenames and provenance.
    raw = {"run": store.run(run_id), "events": store.events(run_id),
           "requests": store.requests(run_id),
           "snapshots": [{"seq": row[0], "state": json.loads(row[1])} for row in
                         store.db.execute("SELECT seq,state FROM snapshots WHERE run_id=? ORDER BY seq",
                                          (run_id,))]}
    (out / "raw.json").write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics = store.metrics(run_id)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    action_frame(store.replay(run_id)).to_csv(out / "actions.csv", index=False)
    pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(store.requests(run_id)).to_csv(out / "requests.csv", index=False)
    return out
