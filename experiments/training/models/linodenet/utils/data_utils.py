from typing import Any, NamedTuple

import torch
from torch import Tensor
from torch import nan as NAN
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# from tsdm.utils.strings import repr_namedtuple


class Inputs(NamedTuple):
    r"""A single sample of the data."""

    t: Tensor
    x: Tensor
    t_target: Tensor


class Sample(NamedTuple):
    r"""A single sample of the data."""

    key: int
    inputs: Inputs
    targets: Tensor
    originals: tuple[Tensor, Tensor]


class Batch(NamedTuple):
    r"""A single sample of the data."""

    x_time: Tensor  # B×N:   the input timestamps.
    x_vals: Tensor  # B×N×D: the input values.
    x_mask: Tensor  # B×N×D: the input mask.

    y_time: Tensor  # B×K:   the target timestamps.
    y_vals: Tensor  # B×K×D: the target values.
    y_mask: Tensor  # B×K×D: teh target mask.


def linodenet_collate(batch: list[Sample]) -> Batch:
    r"""Collate tensors into batch.

    Transform the data slightly: t, x, t_target → T, X where X[t_target:] = NAN
    """
    x_vals: list[Tensor] = []
    y_vals: list[Tensor] = []
    x_time: list[Tensor] = []
    y_time: list[Tensor] = []
    x_mask: list[Tensor] = []
    y_mask: list[Tensor] = []

    for sample in batch:
        t, x, t_target = sample.inputs
        y = sample.targets

        # get whole time interval
        time = torch.cat((t, t_target))
        sorted_idx = torch.argsort(time)

        # pad the x-values
        x_padding = torch.full(
            (t_target.shape[0], x.shape[-1]), fill_value=NAN, device=x.device
        )
        values = torch.cat((x, x_padding))

        # create a mask for looking up the target values
        mask_y = y.isfinite()
        mask_pad = torch.zeros_like(x, dtype=torch.bool)
        mask_x = torch.cat((mask_pad, mask_y))

        x_vals.append(values[sorted_idx])
        x_time.append(time[sorted_idx])
        x_mask.append(mask_x[sorted_idx])

        y_time.append(t_target)
        y_vals.append(y)
        y_mask.append(mask_y)

    return Batch(
        x_time=pad_sequence(x_time, batch_first=True).squeeze(),
        x_vals=pad_sequence(x_vals, batch_first=True, padding_value=NAN).squeeze(),
        x_mask=pad_sequence(x_mask, batch_first=True).squeeze(),
        y_time=pad_sequence(y_time, batch_first=True).squeeze(),
        y_vals=pad_sequence(y_vals, batch_first=True, padding_value=NAN).squeeze(),
        y_mask=pad_sequence(y_mask, batch_first=True).squeeze(),
    )


import numpy as np


def linodenet_collate_sparse(batch: list[Sample]) -> Batch:
    x_time, x_vals, x_mask, y_time, y_vals, y_mask = mimic_collate(batch)
    bs, T, dim = x_vals.shape
    for sample_indx in range(bs):
        torch.manual_seed(np.sum(x_vals.isnan()[sample_indx].detach().numpy()))
        for t in range(T):
            relevant_indexes = torch.argwhere(x_vals[sample_indx, t].isfinite())
            if len(relevant_indexes) > 0:
                keep_ind = relevant_indexes[torch.randint(len(relevant_indexes), (1,))]
                keep_value = x_vals[sample_indx, t, keep_ind].clone()
                x_vals[sample_indx, t, :] = NAN
                x_vals[sample_indx, t, keep_ind] = keep_value

    return Batch(
        x_time=x_time,
        x_vals=x_vals,
        x_mask=x_mask,
        y_time=y_time,
        y_vals=y_vals,
        y_mask=y_mask,
    )


def linodenet_collate_sparse2(batch: list[Sample], sparsity: float) -> Batch:
    r"""Collate tensors into batch.

    Transform the data slightly: t, x, t_target → T, X where X[t_target:] = NAN
    """
    target_x: list[Tensor] = []
    context_x: list[Tensor] = []
    context_vals: list[Tensor] = []
    context_mask: list[Tensor] = []
    target_vals: list[Tensor] = []
    target_mask: list[Tensor] = []

    for sample in batch:
        t, x, t_target = sample.inputs
        y = sample.targets
        # get whole time interval
        # create a mask for looking up the target values
        mask_y = y.isfinite()
        mask_x = x.isfinite()

        mask_x_inds = torch.where(mask_x.sum(-1).bool())
        mask_y_inds = torch.where(mask_y.sum(-1).bool())

        mask_x = mask_x[mask_x_inds]
        mask_y = mask_y[mask_y_inds]
        x = x[mask_x_inds]
        y = y[mask_y_inds]
        t = t[mask_x_inds]
        t_target = t_target[mask_y_inds]

        mask_y_float = mask_y.float()
        mask_x_float = mask_x.float()

        torch.manual_seed(mask_x.sum() + mask_y.sum())

        y_inds = torch.multinomial(mask_y_float, 1)
        x_inds = torch.multinomial(mask_x_float, 1)

        selected_x = torch.zeros_like(mask_x)
        # pdb.set_trace()
        selected_x[
            (torch.arange(0, mask_x.shape[0]).to(x_inds.dtype), x_inds[:, 0])
        ] = True
        inds_x = torch.where(mask_x * ~selected_x)

        indices = torch.randperm(len(inds_x[0]))[
            : int(np.floor(len(inds_x[0]) * sparsity))
        ]
        select_indices = (inds_x[0][indices], inds_x[1][indices])
        selected_x[select_indices] = True

        mask_x = selected_x
        x[~mask_x] = NAN

        selected_y = torch.zeros_like(mask_y)
        if mask_y.shape[0] > 0:
            selected_y[
                (torch.arange(0, mask_y.shape[0]).to(y_inds.dtype), y_inds[:, 0])
            ] = True
            # selected_y[:, y_inds] = True
            inds_y = torch.where(mask_y * ~selected_y)
            indices = torch.randperm(len(inds_y[0]))[
                : int(np.floor(len(inds_y[0]) * sparsity))
            ]
            # if len(indices) == 0:
            select_indices_y = (inds_y[0][indices], inds_y[1][indices])
            selected_y[select_indices_y] = True

            mask_y = selected_y
            y[~mask_y] = NAN

        context_x.append(torch.cat([t, t_target], dim=0))
        x_padding = torch.full(
            (t_target.shape[0], x.shape[-1]), fill_value=NAN, device=x.device
        )
        context_vals.append(torch.cat([x, x_padding], dim=0))
        context_mask.append(torch.cat([torch.zeros_like(x).to(bool), mask_y], dim=0))
        # context_y = torch.cat([context_vals, context_mask], dim=2)
        target_x.append(t_target)
        target_vals.append(y)
        target_mask.append(mask_y)
        # target_y = torch.cat([target_vals, target_mask], dim=2)

    return Batch(
        x_time=pad_sequence(context_x, batch_first=True).squeeze(),
        x_vals=pad_sequence(
            context_vals, batch_first=True, padding_value=NAN
        ).squeeze(),
        x_mask=pad_sequence(context_mask, batch_first=True).squeeze(),
        y_time=pad_sequence(target_x, batch_first=True).squeeze(),
        y_vals=pad_sequence(target_vals, batch_first=True, padding_value=NAN).squeeze(),
        y_mask=pad_sequence(target_mask, batch_first=True).squeeze(),
    )


def mimic_collate(batch: list[Sample]) -> Batch:
    r"""Collate tensors into batch.

    Transform the data slightly: t, x, t_target → T, X where X[t_target:] = NAN
    """
    x_vals: list[Tensor] = []
    y_vals: list[Tensor] = []
    x_time: list[Tensor] = []
    y_time: list[Tensor] = []
    x_mask: list[Tensor] = []
    y_mask: list[Tensor] = []

    for sample in batch:
        t, x, t_target = sample.inputs
        y = sample.targets

        # get whole time interval
        time = torch.cat((t, t_target))
        sorted_idx = torch.argsort(time)

        # pad the x-values
        x_padding = torch.full(
            (t_target.shape[0], x.shape[-1]), fill_value=NAN, device=x.device
        )
        values = torch.cat((x, x_padding))

        # create a mask for looking up the target values
        mask_y = y.isfinite()
        mask_pad = torch.zeros_like(x, dtype=torch.bool)
        mask_x = torch.cat((mask_pad, mask_y))

        x_vals.append(values[sorted_idx])
        x_time.append(time[sorted_idx])
        x_mask.append(mask_x[sorted_idx])

        y_time.append(t_target)
        y_vals.append(y)
        y_mask.append(mask_y)

    return Batch(
        x_time=pad_sequence(x_time, batch_first=True).squeeze(),
        x_vals=pad_sequence(x_vals, batch_first=True, padding_value=NAN).squeeze(),
        x_mask=pad_sequence(x_mask, batch_first=True).squeeze(),
        y_time=pad_sequence(y_time, batch_first=True).squeeze(),
        y_vals=pad_sequence(y_vals, batch_first=True, padding_value=NAN).squeeze(),
        y_mask=pad_sequence(y_mask, batch_first=True).squeeze(),
    )
