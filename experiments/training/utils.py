import glob
import random
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence as pad
from torch.utils.data import DataLoader, Dataset


class Sample(NamedTuple):
    r"""A single sample of the data."""

    key: int
    inputs: tuple
    targets: Tensor


class IMTS_dataset(Dataset):
    def __init__(self, files, ot, fh, fold, sparsity=0.9, noise=0.05):
        # load files
        torch.manual_seed(fold)
        T = []
        X = []
        TY = []
        Y = []
        T_max = max(pd.read_parquet(files[0])["t"])
        value_columns = list(pd.read_parquet(files[0]).columns)
        value_columns.remove("t")
        observation_time = T_max * ot
        forecasting_horizon = observation_time + (T_max * fh)
        for f in files:
            raw_TS = pd.read_parquet(f)
            T.append(raw_TS["t"].loc[raw_TS["t"] <= observation_time].values)
            X.append(raw_TS[value_columns].loc[raw_TS["t"] <= observation_time].values)
            TY.append(
                raw_TS["t"]
                .loc[
                    (raw_TS["t"] > observation_time)
                    & (raw_TS["t"] < forecasting_horizon)
                ]
                .values
            )
            Y.append(
                raw_TS[value_columns]
                .loc[
                    (raw_TS["t"] > observation_time)
                    & (raw_TS["t"] < forecasting_horizon)
                ]
                .values
            )

        T = torch.tensor(np.stack(T, axis=0)).type(torch.float32)
        X = torch.tensor(np.stack(X, axis=0)).type(torch.float32)
        TY = torch.tensor(np.stack(TY, axis=0)).type(torch.float32)
        Y = torch.tensor(np.stack(Y, axis=0)).type(torch.float32)
        # normalize
        T = T / T_max
        TY = TY / T_max
        std_V = torch.std(X.reshape(-1, X.shape[-1]), dim=0)
        mean_V = torch.mean(X.reshape(-1, X.shape[-1]), dim=0)
        X = (X - mean_V) / std_V
        Y = (Y - mean_V) / std_V
        # apply noise
        X += torch.randn(X.shape) * 0.05
        Y += torch.randn(Y.shape) * 0.05
        # apply masking
        M = (torch.rand(X.shape) > sparsity).type(torch.bool)
        MY = (torch.rand(Y.shape) > sparsity).type(torch.bool)
        T_MASK = torch.sum(M, axis=-1) > 0
        TY_MASK = torch.sum(MY, axis=-1) > 0

        T = pad([T[i, T_MASK[i]] for i in range(X.shape[0])], batch_first=True)
        TY = pad([TY[i, TY_MASK[i]] for i in range(X.shape[0])], batch_first=True)
        X = pad([X[i, T_MASK[i], :] for i in range(X.shape[0])], batch_first=True)
        Y = pad([Y[i, TY_MASK[i], :] for i in range(X.shape[0])], batch_first=True)
        M = pad([M[i, T_MASK[i], :] for i in range(X.shape[0])], batch_first=True)
        MY = pad([MY[i, TY_MASK[i], :] for i in range(X.shape[0])], batch_first=True)
        X[~M] = torch.nan
        Y[~MY] = torch.nan

        self.T = T
        self.TY = TY
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return Sample(
            key=idx,
            inputs=(self.T[idx], self.X[idx], self.TY[idx]),
            targets=self.Y[idx],
        )


def create_dataloaders(model, fold, observation_time, forecasting_horizon, sparsity):
    path = f"benchmark_datasets/{model}"
    files = glob.glob(f"{path}/*.parquet")
    random.seed(fold)
    random.shuffle(files)
    train_dataset = IMTS_dataset(
        files=files[: int(len(files) * 0.7)],
        ot=observation_time,
        fh=forecasting_horizon,
        sparsity=sparsity,
        fold=fold,
    )
    valid_dataset = IMTS_dataset(
        files=files[int(len(files) * 0.7) : int(len(files) * 0.8)],
        ot=observation_time,
        fh=forecasting_horizon,
        sparsity=sparsity,
        fold=fold,
    )
    test_dataset = IMTS_dataset(
        files=files[int(len(files) * 0.8) :],
        ot=observation_time,
        fh=forecasting_horizon,
        sparsity=sparsity,
        fold=fold,
    )
    out_path = f"IMTS_benchmark_datasets/{sparsity}/{model}/{fold}"
    torch.save(train_dataset, f"{out_path}/train.pt")
    torch.save(valid_dataset, f"{out_path}/valid.pt")
    torch.save(test_dataset, f"{out_path}/test.pt")


def get_data_loaders(path, fold, batch_size, collate_fn):
    train_dataset = torch.load(f"{path}/{fold}/train.pt", weights_only=False)
    valid_dataset = torch.load(f"{path}/{fold}/valid.pt", weights_only=False)
    test_dataset = torch.load(f"{path}/{fold}/test.pt", weights_only=False)

    TRAIN_LOADER = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    VALID_LOADER = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    TEST_LOADER = DataLoader(
        test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
    )
    return TRAIN_LOADER, VALID_LOADER, TEST_LOADER
