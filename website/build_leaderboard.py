"""Build the leaderboard data for the website.

Inputs
    website/models.json       model metadata; "slot" fixes each model's color (1-8, 0 = neutral gray)
    website/datasets.csv      short dataset ids (DUP01, ...) for the 50 benchmark datasets
    website/results/*.csv     one file per model, either
                                dataset,fold,test_mse        (one row per dataset and fold), or
                                dataset,mse_mean,mse_std     (already aggregated)
    resources/Top50_final.csv dataset metadata (domain, JGD, channels, ...)

Output
    docs/data/results.json    everything the page needs; all statistics are computed in the browser

Usage
    python website/build_leaderboard.py

Only the standard library is used, so this runs anywhere (including CI) without the training environment.
"""

import argparse
import csv
import datetime
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "website"
REPO_URL = "https://github.com/kloetergensc/Physiome-ODE"
STATUSES = {"paper", "verified", "self-reported"}
N_FOLDS = 5
MAX_SLOT = 8  # validated categorical colors on the page; slot 0 is neutral gray


def warn(msg):
    print(f"warning: {msg}", file=sys.stderr)


def load_datasets():
    ids = {r["dataset"]: r["id"] for r in csv.DictReader(open(WEB / "datasets.csv"))}
    datasets = []
    for r in csv.DictReader(open(ROOT / "resources" / "Top50_final.csv")):
        if r["model"] not in ids:
            raise SystemExit(f"datasets.csv has no id for {r['model']}")
        datasets.append({
            "id": ids[r["model"]],
            "model": r["model"],
            "domain": r["domain"],
            "jgd": float(r["high_score"]),
            "channels": int(r["channel"]),
            "constants": int(r["constants"]),
            "T": float(r["T"]),
            "ds": float(r["d_states"]),
            "dc": float(r["d_constants"]),
            "unit": r["Time Unit"],
        })
    datasets.sort(key=lambda d: -d["jgd"])
    return datasets


def load_results(path, names):
    rows = list(csv.DictReader(open(path)))
    cols = set(rows[0]) if rows else set()
    out = {}
    if {"dataset", "fold", "test_mse"} <= cols:
        folds = defaultdict(list)
        for r in rows:
            folds[r["dataset"]].append(float(r["test_mse"]))
        for d, v in folds.items():
            if len(v) != N_FOLDS:
                warn(f"{path.name}: {d} has {len(v)} folds, expected {N_FOLDS}")
            sd = statistics.stdev(v) if len(v) > 1 else 0.0
            out[d] = [float(f"{statistics.fmean(v):.6g}"), float(f"{sd:.6g}")]
    elif {"dataset", "mse_mean", "mse_std"} <= cols:
        for r in rows:
            out[r["dataset"]] = [float(r["mse_mean"]), float(r["mse_std"])]
    else:
        raise SystemExit(f"{path}: expected columns dataset,fold,test_mse or dataset,mse_mean,mse_std")
    unknown = sorted(set(out) - names)
    if unknown:
        raise SystemExit(f"{path}: unknown datasets {unknown}")
    return out


def build():
    cfg = json.load(open(WEB / "models.json"))
    datasets = load_datasets()
    names = {d["model"] for d in datasets}
    models, results = [], {}
    for m in cfg["models"]:
        if m["status"] not in STATUSES:
            raise SystemExit(f"{m['name']}: status must be one of {sorted(STATUSES)}")
        res = load_results(WEB / "results" / m["results"], names)
        missing = sorted(names - set(res))
        if missing:
            warn(f"{m['name']} is missing {len(missing)} of {len(names)} datasets and is left out: {missing[:5]}...")
            continue
        kind = "blob" if m["code"].endswith(".py") else "tree"
        models.append({
            "name": m["name"],
            "family": m["family"],
            "ref": m["reference"],
            "code": f"{REPO_URL}/{kind}/main/{m['code']}",
            "status": m["status"],
            "source": m.get("source", ""),
            "slot": m["slot"],
        })
        results[m["name"]] = res
    if cfg["anchor"] not in results:
        raise SystemExit(f"anchor model {cfg['anchor']} has no complete results")
    slots = [m["slot"] for m in cfg["models"]]
    taken = [s for s in slots if s != 0]
    if any(not isinstance(s, int) or not 0 <= s <= MAX_SLOT for s in slots) or len(taken) != len(set(taken)):
        raise SystemExit(f"models.json: each slot must be a distinct integer 1-{MAX_SLOT}, or 0 for neutral gray")
    for d in datasets:
        d["res"] = {m["name"]: results[m["name"]][d["model"]] for m in models}
    return {
        "generated": datetime.date.today().isoformat(),
        "anchor": cfg["anchor"],
        "models": models,
        "datasets": datasets,
    }


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    data = build()
    out = ROOT / "docs" / "data" / "results.json"
    out.write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    print(f"wrote {out.relative_to(ROOT)}: {len(data['models'])} models, {len(data['datasets'])} datasets")


if __name__ == "__main__":
    main()
