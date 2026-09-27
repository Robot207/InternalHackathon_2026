from __future__ import annotations

import datetime as dt
import json
import sys
import time

from backend import artifacts
from backend.dataset import build_dataset, load_dataset
from backend.training import run_training


def main() -> None:
    preset = sys.argv[1] if len(sys.argv) > 1 else "london"
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    model = sys.argv[3] if len(sys.argv) > 3 else "random_forest"
    end = (dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=1)).isoformat()
    start = (dt.date.fromisoformat(end) - dt.timedelta(days=days - 1)).isoformat()

    t0 = time.time()
    last = ["", 0.0]

    def progress(p: float, stage: str) -> None:
        if stage != last[0] or p - last[1] > 0.2:
            print(f"  [{p:5.0%}] {stage}", flush=True)
            last[0], last[1] = stage, p

    print(f"dataset: preset={preset} {start}..{end}")
    summary = build_dataset(preset, start, end, 0.05, progress=progress)
    print(
        f"  cells={summary['n_lat']}x{summary['n_lon']} hours={summary['n_times']} "
        f"gap={summary['gap_fraction']:.1%} ref={summary['has_reference']} "
        f"warnings={summary['warnings']}"
    )
    data = load_dataset(summary["key"])
    meta = run_training(data, model, "spatiotemporal", True, summary, progress)
    artifacts.write_meta(meta)
    artifacts.write_layers(summary)
    print(json.dumps(meta["metrics"], indent=2))
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
