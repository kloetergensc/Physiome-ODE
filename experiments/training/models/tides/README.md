# TIDES

TIDES (Time-aware Input-Dependent State-space) from

> Taylan Soydan, Miguel A. Bessa, Dirk Mohr, Rui Barreira.
> *TIDES: Implicit Time-Awareness in Selective State Space Models.* NeurIPS 2026.
> [arXiv:2605.09742](https://arxiv.org/abs/2605.09742)

This folder holds the authors' implementation, taken from
[github.com/TaylanSoydan/TIDES](https://github.com/TaylanSoydan/TIDES) at commit
[`400aa0c`](https://github.com/TaylanSoydan/TIDES/tree/400aa0cc41d53c88bdcf816a5ee4c1538b90e3cd)
(2026-10-08). It is distributed under the MIT License, Copyright (c) 2026 Taylan Soydan; see `LICENSE`.
Please cite the paper if you use it, and report problems with the model upstream.

| File here | Upstream file | Changes |
|:--|:--|:--|
| `__init__.py`, `tides.py`, `tides_collate.py`, `tides_forecasting.py`, `_hub.py` | `tides/` | none |
| `tides_train_fn.py` | `physiome_ode/tides_train_fn.py` | the two package imports made relative; provenance note in the docstring |
| `configs/winners.csv` | `physiome_ode/configs/winners.csv` | none |
| `../../train_tides.py` | `physiome_ode/run_final_folds.py` | import paths, location of `winners.csv`, default data path `../../data/final`, `-dset` alias for `--dataset` |
| `LICENSE` | `LICENSE` | none |
| `LICENSE-S5` | [`LICENSE`](https://github.com/lindermanlab/S5/blob/d133f4d/LICENSE) of S5, not in the TIDES repository | none |

Upstream's `physiome_ode/utils.py` is this repository's `experiments/training/utils.py` with docstrings added
(the code is identical), so the training code imports ours instead.

`tides.py` adapts SSM utilities (associative scan, discretization, HiPPO initialization) from
[S5](https://github.com/lindermanlab/S5) (Smith, Warrington and Linderman, ICLR 2023), as its header notes.
S5 is under the MIT License, Copyright (c) 2022 Linderman Lab; its license is included here as `LICENSE-S5`.
The header of `tides.py`, left unchanged, gives S5's license as Apache-2.0, but S5's own `LICENSE` is MIT.

## Running

From `experiments/training`, with the data in `../../data/final/<dataset>/<fold>/`:

```
python train_tides.py --winner HOD01              # the paper's configuration for HOD01, all 5 folds
python train_tides.py --winner HOD01 --folds 0    # one fold
python train_tides.py -dset hodgkin_huxley_1952_variant01 --hidden_size 32 --ssm_size 8 \
    --ssm_blocks 2 --num_blocks 3                 # hyperparameters by hand
```

`configs/winners.csv` holds the per-dataset hyperparameters behind the paper's Physiome-ODE results
(Table 8 there; on the leaderboard as `website/results/tides.csv`). Training runs up to 200 epochs with early
stopping on validation MSE (patience 30), and fold k uses seed k. It needs `torch`, `numpy`, `scipy` and
`tqdm`; `huggingface_hub` is only needed for `save_pretrained` / `from_pretrained`, and `wandb` only with
`--wandb_mode`.

The script prints the mean and `np.std` (population standard deviation) over folds. For fold-level result
files, `website/build_leaderboard.py` reports the sample standard deviation instead.

## Citation

```bibtex
@inproceedings{soydan2026tides,
  title     = {{TIDES}: Implicit Time-Awareness in Selective State Space Models},
  author    = {Soydan, Taylan and Bessa, Miguel A. and Mohr, Dirk and Barreira, Rui},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  note      = {arXiv:2605.09742}
}
```
