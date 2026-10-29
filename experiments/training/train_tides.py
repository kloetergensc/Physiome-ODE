"""
Train and test TIDES on all 5 folds of one Physiome-ODE dataset.
Reports mean ± std test MSE, directly comparable to the Physiome-ODE leaderboard.

From physiome_ode/run_final_folds.py of https://github.com/TaylanSoydan/TIDES
(commit 400aa0c), Copyright (c) 2026 Taylan Soydan, MIT License (see
models/tides/LICENSE). Changed here: the import paths, the location of
winners.csv, the default data path (../../data/final, as for the other models)
and a -dset alias for --dataset. See models/tides/README.md.

Reproduce a paper number (configs in models/tides/configs/winners.csv, one per
dataset; see docs/reproducibility.md in the TIDES repository):

    python train_tides.py --winner HOD01
    python train_tides.py --winner hodgkin_huxley_1952_variant01

Or with hyperparameters given by hand (any flag also overrides --winner):

    python train_tides.py \\
        --dataset hodgkin_huxley_1952_variant01 \\
        --hidden_size 32 --ssm_size 8 --ssm_blocks 2 --num_blocks 3 \\
        --lr 5e-4 --weight_decay 1e-4 --batch_size 32

Folds use seed = fold index; training runs up to 200 epochs with early stopping
on validation MSE (patience 30), as in the paper.
"""

import argparse
import csv
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from models.tides.tides_train_fn import train_tides  # noqa: E402
from utils import IMTS_dataset  # noqa: E402,F401 — pickle deserialization

WINNERS_CSV = os.path.join(_HERE, "models", "tides", "configs", "winners.csv")
WANDB_PROJECT = "tides"

# train_tides keyword -> type, for every hyperparameter a config can set
HP_TYPES = {
    "hidden_size": int, "ssm_size": int, "ssm_blocks": int, "num_blocks": int,
    "encoder_depth": int, "lambda_encoder_depth": int, "bc_rank": int,
    "lambda_re_mode": str, "lambda_im_mode": str, "bc_mode": str,
    "learn_lambda": str, "discretization": str, "proj_init_method": str,
    "proj_norm": str, "drop_rate": float, "dt_min": float, "ff_mult": float,
    "bidir": bool, "clip_eigs": bool,
    "lr": float, "lr_factor": int, "warmup_epochs": int,
    "weight_decay": float, "batch_size": int,
}
DEFAULTS = {
    "encoder_depth": 0, "lambda_encoder_depth": 0, "bc_rank": 8,
    "lambda_re_mode": "input_dependent", "lambda_im_mode": "lti",
    "bc_mode": "input_dependent", "learn_lambda": "standard",
    "discretization": "zoh", "proj_init_method": "zeros", "proj_norm": "rmsnorm",
    "drop_rate": 0.0, "dt_min": 0.001, "ff_mult": 1.0, "bidir": False,
    "clip_eigs": False, "lr": 1e-3, "lr_factor": 1,
    "warmup_epochs": 0, "weight_decay": 1e-3, "batch_size": 32,
}


def _cast(key, value):
    kind = HP_TYPES[key]
    if kind is bool:
        return value if isinstance(value, bool) else str(value).lower() in ("true", "1", "yes")
    if key == "proj_norm" and str(value).lower() in ("", "none"):
        return None
    return kind(float(value)) if kind is int else kind(value)


def load_winner(name: str) -> tuple:
    """(dataset, hyperparameters) for a dataset code (HOD01) or name from winners.csv."""
    with open(WINNERS_CSV) as f:
        for row in csv.DictReader(f):
            if name in (row["code"], row["dataset"]):
                return row["dataset"], {k: _cast(k, row[k]) for k in HP_TYPES if k in row}
    raise SystemExit(f"{name!r} is not a code or dataset name in {WINNERS_CSV}")


def main():
    parser = argparse.ArgumentParser(
        description="Run 5 folds of TIDES on one Physiome-ODE dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--winner", default=None,
                        help="Dataset code (e.g. HOD01) or name: use its row of configs/winners.csv")
    parser.add_argument("-dset", "--dataset", default=None, type=str)
    parser.add_argument("--data_base_path", default=os.path.join(_HERE, "..", "..", "data", "final"),
                        help="Directory holding <dataset>/<fold>/ (the Zenodo 'final/' folder)")
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--early_stop", default=30, type=int)
    parser.add_argument("--folds", default=[0, 1, 2, 3, 4], type=int, nargs="+")
    parser.add_argument("--saved_models_dir", default="saved_models", type=str)
    parser.add_argument("--wandb_mode", default="disabled", choices=["disabled", "offline", "online"],
                        help="Weights & Biases logging (off unless asked for)")
    # Manual hyperparameters; any flag given overrides --winner
    for key, kind in HP_TYPES.items():
        if kind is bool:
            parser.add_argument(f"--{key}", default=None, type=lambda v: _cast("bidir", v),
                                metavar="{true,false}")
        else:
            parser.add_argument(f"--{key}", default=None, type=kind if kind is not int else float)
    args = parser.parse_args()

    # ── Collect hyperparameters: defaults < winner < explicit flags ──
    hp = dict(DEFAULTS)
    dataset = args.dataset
    if args.winner:
        dataset, winner_hp = load_winner(args.winner)
        hp.update(winner_hp)
        print(f"Using the winning configuration for {args.winner} from {WINNERS_CSV}")
    if args.dataset:
        dataset = args.dataset
    if dataset is None:
        parser.error("give --winner, or --dataset with the hyperparameters as flags")
    for key in HP_TYPES:
        if getattr(args, key) is not None:
            hp[key] = _cast(key, getattr(args, key))
    missing = [k for k in ("hidden_size", "ssm_size", "ssm_blocks", "num_blocks") if k not in hp]
    if missing:
        parser.error(f"missing hyperparameters: {missing}")

    wandb = None
    if args.wandb_mode != "disabled":
        import wandb

    print(f"\nDataset : {dataset}")
    print(f"Epochs  : {args.epochs}  early_stop={args.early_stop}  folds={args.folds}")
    print("Config  : " + "  ".join(f"{k}={hp[k]}" for k in sorted(hp)))
    print()

    val_losses, test_losses, test_maes = [], [], []
    for fold in args.folds:
        print(f"── Fold {fold} ──────────────────────────────────")
        run = None
        if wandb is not None:
            run = wandb.init(project=WANDB_PROJECT, name=f"{dataset}_final_f{fold}",
                             config={**hp, "dataset": dataset, "fold": fold,
                                     "epochs": args.epochs, "phase": "final_eval"},
                             tags=[dataset, "final_eval"], reinit=True, mode=args.wandb_mode)
        result = train_tides(
            dataset=dataset, fold=fold, data_base_path=args.data_base_path,
            epochs=args.epochs, early_stop_patience=args.early_stop, seed=fold,
            saved_models_dir=args.saved_models_dir, verbose=True, wandb_run=run, **hp)
        if run is not None:
            wandb.log({"val_loss": result["val_loss"], "test_loss": result["test_loss"],
                       "test_mae": result["test_mae"], "fold": fold})
            wandb.finish()
        val_losses.append(result["val_loss"])
        test_losses.append(result["test_loss"])
        test_maes.append(result["test_mae"])
        print(f"  fold {fold}: val={result['val_loss']:.4f}  "
              f"test_loss={result['test_loss']:.4f}  test_mae={result['test_mae']:.4f}  "
              f"params={result['num_params']:,}  stopped_early={result['stopped_early']}")

    print()
    print("=" * 60)
    print(f"Dataset: {dataset}   folds: {args.folds}")
    print(f"best_val_loss: {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
    print(f"test_loss:     {np.mean(test_losses):.4f} ± {np.std(test_losses):.4f}   ← leaderboard number")
    print(f"test_mae:      {np.mean(test_maes):.4f} ± {np.std(test_maes):.4f}")
    print(f"Number of trainable parameters: {result['num_params']:,}")


if __name__ == "__main__":
    main()
