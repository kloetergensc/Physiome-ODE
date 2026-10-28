"""Regenerate example instances of every benchmark dataset for the website's sample viewer.

Instances are produced exactly like data_creation/generate.py does for the final benchmark
(generate_benchmark.py: T/2, N_t=100, random start, same seeds and lsoda settings), so instance n
here is instance n of the released data, up to solver and library versions. Values are z-scored
with the observation-window statistics of the generated instances; the page adds the benchmark's
noise and 20% sampling mask in the browser on every "Sample" click.

Usage (from the repository root, in the training environment):
    python website/generate_samples.py                 # all 50 datasets, 12 instances each
    python website/generate_samples.py --only LEN01    # a single dataset
"""

import sys

sys.dont_write_bytecode = True  # importing the Physiome model.py files must not rewrite their __pycache__

import argparse
import csv
import importlib.util
import json
import re
import time
import warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from scipy.integrate import ode

ROOT = Path(__file__).resolve().parent.parent
TOTAL_SAMPLES = 2000  # total_samples in generate_benchmark.py; sets the seed offset for retries
N_T = 100


def generate(task):
    domain, model, T, d_states, d_constants, n = task
    warnings.filterwarnings("error")
    seed = n
    for _ in range(10):
        np.random.seed(seed)
        spec = importlib.util.spec_from_file_location("model_module", ROOT / "models" / domain / model / "model.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        legend_states, _, _, _ = module.createLegends()
        init_states, constants = module.initConsts()
        init_states = [np.random.uniform(x * (1 - d_states), x * (1 + d_states)) for x in init_states]
        constants = [np.random.uniform(x * (1 - d_constants), x * (1 + d_constants)) for x in constants]
        voi = np.linspace(0, T * 2, N_T * 2)
        try:
            r = ode(module.computeRates)
            r.set_integrator("lsoda", method="bdf", atol=1e-6, rtol=1e-6, max_step=1)
            r.set_initial_value(init_states, voi[0])
            r.set_f_params(constants)
            states = np.zeros((module.sizeStates, len(voi)))
            states[:, 0] = init_states
            for i, t in enumerate(voi[1:]):
                if r.successful():
                    r.integrate(t)
                    states[:, i + 1] = r.y
                else:
                    break
        except (Exception, RuntimeWarning, UserWarning):
            seed += TOTAL_SAMPLES
            continue
        start = np.random.randint(0, N_T)
        X = states[:, start:start + N_T].T
        if np.max(np.abs(X)) > 1e15 or np.isnan(X).any():
            seed += TOTAL_SAMPLES
            continue
        return task, voi[:N_T], X, list(legend_states)
    return task, None, None, None


def parse_legend(s):
    m = re.match(r"^(.*?) in component (.*?) \((.*)\)$", s)
    return {"name": m.group(1), "comp": m.group(2), "unit": m.group(3)} if m else {"name": s, "comp": "", "unit": ""}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances", type=int, default=12, help="instances per dataset (default 12)")
    ap.add_argument("--only", nargs="*", help="dataset ids to regenerate, e.g. LEN01 DUP02")
    ap.add_argument("--workers", type=int, default=cpu_count())
    args = ap.parse_args()

    ids = {r["dataset"]: r["id"] for r in csv.DictReader(open(ROOT / "website" / "datasets.csv"))}
    rows = [r for r in csv.DictReader(open(ROOT / "resources" / "Top50_final.csv"))
            if not args.only or ids[r["model"]] in args.only]
    tasks = [(r["domain"], r["model"], float(r["T"]) / 2, float(r["d_states"]), float(r["d_constants"]), n)
             for r in rows for n in range(args.instances)]
    out_dir = ROOT / "docs" / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    generated = {}
    with Pool(args.workers) as pool:
        for task, t, X, legend in pool.imap_unordered(generate, tasks):
            generated.setdefault(task[1], []).append((task[5], t, X, legend))

    for r in rows:
        ok = sorted((g for g in generated[r["model"]] if g[2] is not None), key=lambda g: g[0])
        if not ok:
            print(f"FAILED {r['model']}", file=sys.stderr)
            continue
        t = ok[0][1]
        X = np.stack([g[2] for g in ok])  # instances x time x channels
        obs = t <= t.max() * 0.5
        mean = X[:, obs].reshape(-1, X.shape[-1]).mean(0)
        std = X[:, obs].reshape(-1, X.shape[-1]).std(0)
        std[std == 0] = 1
        Z = (X - mean) / std
        unit = re.search(r"\((.*)\)", r["Time Unit"])
        out = {
            "id": ids[r["model"]],
            "model": r["model"],
            "domain": r["domain"],
            "timeUnit": unit.group(1) if unit else "",
            "t": [round(float(x), 4) for x in t],
            "channels": [dict(parse_legend(s), score=round(sc, 5))
                         for s, sc in zip(ok[0][3], json.loads(r["channel_scores"]))],
            "instances": [g[0] for g in ok],
            "z": [[[round(float(v), 2) for v in Z[k, :, c]] for c in range(Z.shape[-1])] for k in range(len(ok))],
        }
        (out_dir / f"{out['id']}.json").write_text(json.dumps(out, separators=(",", ":")))
        print(f"{out['id']:6s} {r['model']}: {len(ok)} instances, {Z.shape[-1]} channels")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
