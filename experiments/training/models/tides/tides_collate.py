"""TIDES collate function for the Physiome-ODE benchmark.

Converts IMTS_dataset samples (irregularly-sampled obs + targets) into
TIDES-compatible batches with a merged sorted timeline and per-step delta_t.

Strategy (mirrors how GraFITi merges T+TY, adds step_scale for SSM):
  1. Merge observation timestamps T and target timestamps TY into one sorted timeline
  2. Compute delta_t between consecutive steps → step_scale for TIDES discretization
  3. Fill values at observation positions (zeros where M=False), zeros at target positions
  4. Track which positions are targets (for loss extraction)
"""

from typing import NamedTuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


class TIDESBatch(NamedTuple):
    """Batch format consumed by TIDESForecastingModel (see physiome_ode/tides_train_fn.py)."""

    values:      Tensor  # (B, L, D)  merged timeline values; zeros at target positions
    step_scale:  Tensor  # (B, L)     delta_t between consecutive merged timestamps
    target_mask: Tensor  # (B, L, D)  True at target positions where MY=True (valid targets)
    y_vals:      Tensor  # (B, L, D)  ground-truth values; zeros at observation positions


def tides_collate(batch) -> TIDESBatch:
    """Collate IMTS_dataset samples into an TIDESBatch.

    Each sample is Sample(key, inputs=(t, x, t_target), targets=y) where:
      - t:        (N,)    observation timestamps in [0, 1]  (normalized by IMTS_dataset)
      - x:        (N, D)  observation values; NaN where M=False (unobserved)
      - t_target: (K,)    target timestamps in [0, 1]
      - y:        (K, D)  target values;   NaN where MY=False

    Returns TIDESBatch with padded tensors of shape (B, L, ...) where
    L = max(N_i + K_i) across the batch.
    """
    values_list:      list[Tensor] = []
    step_scale_list:  list[Tensor] = []
    target_mask_list: list[Tensor] = []
    y_vals_list:      list[Tensor] = []

    for sample in batch:
        t, x, t_target = sample.inputs   # (N,), (N, D), (K,)
        y = sample.targets                # (K, D)

        N = t.shape[0]
        K = t_target.shape[0]
        D = x.shape[-1]

        # Masks: NaN marks unobserved entries (set by IMTS_dataset lines 81-82)
        mask_x = x.isfinite()   # (N, D) — True where observation exists
        mask_y = y.isfinite()   # (K, D) — True where target is valid

        x = x.nan_to_num(0.0)
        y = y.nan_to_num(0.0)

        # ── Merge and sort obs + target timestamps ────────────────────────────
        merged_t = torch.cat([t, t_target], dim=0)   # (N+K,)
        sort_idx = merged_t.argsort()
        sorted_t = merged_t[sort_idx]                 # (N+K,) ascending

        # ── Step scale: [t_first, Δt_1, Δt_2, ..., Δt_{L-1}] ────────────────
        # Prepend first timestamp so every position has a step value.
        # Timestamps already in [0, 1], so step_scale ∈ [0, 1] as well.
        delta = torch.diff(sorted_t)                      # (N+K-1,)
        step_s = torch.cat([sorted_t[:1], delta])         # (N+K,)

        # ── Boolean map: which sorted positions are targets ───────────────────
        is_target = sort_idx >= N                          # (N+K,)

        # ── Values: obs values at obs positions, zeros at target positions ─────
        vals = torch.zeros(N + K, D)
        obs_positions = ~is_target
        orig_obs_idx = sort_idx[obs_positions]             # indices into t / x (0..N-1)
        # x already zero-filled at unobserved positions (nan_to_num above),
        # so mask_x is not needed here — zeros propagate naturally.
        vals[obs_positions] = x[orig_obs_idx]

        # ── Target mask: True at target positions with valid Y ─────────────────
        tgt_mask = torch.zeros(N + K, D, dtype=torch.bool)
        orig_tgt_idx = sort_idx[is_target] - N            # indices into t_target / y (0..K-1)
        tgt_mask[is_target] = mask_y[orig_tgt_idx]

        # ── Target values (for loss) ──────────────────────────────────────────
        tgt_vals = torch.zeros(N + K, D)
        tgt_vals[is_target] = y[orig_tgt_idx]

        values_list.append(vals)
        step_scale_list.append(step_s)
        target_mask_list.append(tgt_mask)
        y_vals_list.append(tgt_vals)

    return TIDESBatch(
        values=pad_sequence(
            values_list, batch_first=True, padding_value=0.0
        ),           # (B, L, D)
        step_scale=pad_sequence(
            step_scale_list, batch_first=True, padding_value=0.0
        ),           # (B, L)
        target_mask=pad_sequence(
            target_mask_list, batch_first=True, padding_value=False
        ),           # (B, L, D)
        y_vals=pad_sequence(
            y_vals_list, batch_first=True, padding_value=0.0
        ),           # (B, L, D)
    )
