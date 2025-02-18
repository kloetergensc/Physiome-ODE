import argparse
import logging
import os
import pdb
import random
import sys
import time
import warnings
from datetime import datetime
from random import SystemRandom

import numpy as np
import torch
from IPython.core.display import HTML
from models.grafiti.gratif import tsdm_collate
from torch import Tensor, jit
from utils import IMTS_dataset, get_data_loaders

# fmt: off
parser = argparse.ArgumentParser(description="Training Script for USHCN dataset.")
parser.add_argument("-q",  "--quiet",        default=False,  const=True, help="kernel-inititialization", nargs="?")
parser.add_argument("-r",  "--run_id",       default=None,   type=str,   help="run_id")
parser.add_argument("-c",  "--config",       default=None,   type=str,   help="load external config", nargs=2)
parser.add_argument("-e",  "--epochs",       default=45,    type=int,   help="maximum epochs")
parser.add_argument("-es",  "--early-stop",  default=10,    type=int,   help="early stop patience")
parser.add_argument("-f",  "--fold",         default=0,      type=int,   help="fold number")
parser.add_argument("-bs", "--batch-size",   default=32,     type=int,   help="batch-size")
parser.add_argument("-lr", "--learn-rate",   default=0.001,  type=float, help="learn-rate")
parser.add_argument("-b",  "--betas", default=(0.9, 0.999),  type=float, help="adam betas", nargs=2)
parser.add_argument("-wd", "--weight-decay", default=0.001,  type=float, help="weight-decay")
parser.add_argument("-hs", "--hidden-size",  default=32,    type=int,   help="hidden-size")
parser.add_argument("-ki", "--kernel-init",  default="skew-symmetric",   help="kernel-inititialization")
parser.add_argument("-n",  "--note",         default="",     type=str,   help="Note that can be added")
parser.add_argument("-nl",  "--nlayers", default=2,   type=int,   help="")
parser.add_argument("-ahd",  "--attn-head", default=2,   type=int,   help="")
parser.add_argument("-ldim",  "--latent-dim", default=128,   type=int,   help="")
parser.add_argument("-fh", "--forc-time", default=0.5, type=float, help="forecast horizon [0,1]")
parser.add_argument("-ot", "--observation-time", default=0.5, type=float, help="conditioning range [0,1]")
parser.add_argument("-nf", "--nfolds", default=5, type=int, help="#folds for crossvalidation")
parser.add_argument("-dset", "--dataset", required=True, type=str, help="Name of the dataset")
import pdb

# fmt: on


ARGS = parser.parse_args()
print(" ".join(sys.argv))
experiment_id = int(SystemRandom().random() * 10000000)
print(ARGS, experiment_id)

torch.manual_seed(ARGS.fold)
random.seed(ARGS.fold)
np.random.seed(ARGS.fold)


import yaml

if ARGS.config is not None:
    cfg_file, cfg_id = ARGS.config
    with open(cfg_file, "r") as file:
        cfg_dict = yaml.safe_load(file)
        vars(ARGS).update(**cfg_dict[int(cfg_id)])

print(ARGS)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

model_path = (
    "saved_models/" + "GrAFITi_" + ARGS.dataset + "_" + str(experiment_id) + ".h5"
)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

warnings.filterwarnings(action="ignore", category=UserWarning, module="torch")
logging.basicConfig(level=logging.WARN)
HTML("<style>.jp-OutputArea-prompt:empty {padding: 0; border: 0;}</style>")


OPTIMIZER_CONFIG = {
    "lr": ARGS.learn_rate,
    "betas": ARGS.betas,
    "weight_decay": ARGS.weight_decay,
}

TRAIN_LOADER, VALID_LOADER, TEST_LOADER = get_data_loaders(
    fold=ARGS.fold,
    path=f"../../data/final/{ARGS.dataset}/",
    observation_time=ARGS.observation_time,
    forecasting_horizon=ARGS.forc_time,
    batch_size=ARGS.batch_size,
    collate_fn=tsdm_collate,
)
EVAL_LOADERS = {"train": TRAIN_LOADER, "valid": VALID_LOADER, "test": TEST_LOADER}


def MSE(y: Tensor, yhat: Tensor, mask: Tensor) -> Tensor:
    err = torch.mean((y[mask] - yhat[mask]) ** 2)
    return err


def MAE(y: Tensor, yhat: Tensor, mask: Tensor) -> Tensor:
    err = torch.sum(mask * torch.abs(y - yhat), 1) / (torch.sum(mask, 1))
    return torch.mean(err)


def RMSE(y: Tensor, yhat: Tensor, mask: Tensor) -> Tensor:
    err = torch.sqrt(torch.sum(mask * (y - yhat) ** 2, 1) / (torch.sum(mask, 1)))
    return torch.mean(err)


METRICS = {
    "RMSE": jit.script(RMSE),
    "MSE": jit.script(MSE),
    "MAE": jit.script(MAE),
}
LOSS = jit.script(MSE)


from models.grafiti.gratif import GrATiF

batch = next(iter(TRAIN_LOADER))
T, X, M, TY, Y, MY = (tensor for tensor in batch)

MODEL_CONFIG = {
    "input_dim": X.shape[-1],
    "attn_head": ARGS.attn_head,
    "latent_dim": ARGS.latent_dim,
    "n_layers": ARGS.nlayers,
    "device": DEVICE,
}

MODEL = GrATiF(**MODEL_CONFIG).to(DEVICE)


def predict_fn(model, batch) -> tuple[Tensor, Tensor, Tensor]:
    """Get targets and predictions."""
    T, X, M, TY, Y, MY = (tensor.to(DEVICE) for tensor in batch)
    output, target_U_, target_mask_ = model(T, X, M, TY, Y, MY)
    return target_U_, output.squeeze(), target_mask_


MODEL.zero_grad(set_to_none=True)


# Forward
Y, YHAT, MASK = predict_fn(MODEL, batch)
# Backward
R = LOSS(Y, YHAT, MASK)
assert torch.isfinite(R).item(), "Model Collapsed!"
# R.backward()

# Reset
MODEL.zero_grad(set_to_none=True)

# ## Initialize Optimizer

from torch.optim import AdamW

OPTIMIZER = AdamW(MODEL.parameters(), **OPTIMIZER_CONFIG)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    OPTIMIZER, "min", patience=10, factor=0.5, min_lr=0.00001, verbose=True
)
ovr_start_time = time.time()
es = False
best_val_loss = 10e8
total_num_batches = 0
epoch_times = []
for epoch in range(1, ARGS.epochs + 1):
    loss_list = []
    start_time = time.time()
    for batch in TRAIN_LOADER:
        total_num_batches += 1
        OPTIMIZER.zero_grad()
        Y, YHAT, MASK = predict_fn(MODEL, batch)
        R = LOSS(Y, YHAT, MASK)
        assert torch.isfinite(R).item(), "Model Collapsed!"
        loss_list.append([R])
        # Backward
        R.backward()
        OPTIMIZER.step()
    epoch_time = time.time()
    train_loss = torch.mean(torch.Tensor(loss_list))
    loss_list = []
    count = 0
    with torch.no_grad():
        for batch in VALID_LOADER:
            total_num_batches += 1
            # Forward
            Y, YHAT, MASK = predict_fn(MODEL, batch)
            R = LOSS(Y, YHAT, MASK)
            if R.isnan():
                pdb.set_trace()
            loss_list.append([R * MASK.sum()])
            count += MASK.sum()
    val_loss = torch.sum(torch.Tensor(loss_list).to(DEVICE) / count)
    epoch_times.append(epoch_time - start_time)
    print(
        f"{epoch:3.0f} Train:{train_loss.item():4.4f}  VAL: {val_loss.item():4.4f}   EPOCH_TIME: {epoch_times[-1]:3.2f}"
    )
    if val_loss < best_val_loss:
        best_val_loss = val_loss

        torch.save(
            {
                "args": ARGS,
                "epoch": epoch,
                "state_dict": MODEL.state_dict(),
                "optimizer_state_dict": OPTIMIZER.state_dict(),
                "loss": train_loss,
            },
            model_path,
        )
        early_stop = 0
    else:
        early_stop += 1
    if early_stop == ARGS.early_stop:
        print(
            f"Early stopping because of no improvement in val. metric for {ARGS.early_stop} epochs"
        )
        es = True
    scheduler.step(val_loss)

    # LOGGER.log_epoch_end(epoch)
    if (epoch == ARGS.epochs) or (es == True):
        print(f"tot_train_time: {time.time()- ovr_start_time}")
        chp = torch.load(model_path)
        MODEL.load_state_dict(chp["state_dict"])
        loss_list = []
        mae_list = []
        count = 0
        with torch.no_grad():
            inf_start = time.time()
            for batch in TEST_LOADER:
                total_num_batches += 1
                # Forward
                Y, YHAT, MASK = predict_fn(MODEL, batch)
                R = LOSS(Y, YHAT, MASK)
                assert torch.isfinite(R).item(), "Model Collapsed!"
                # loss_list.append([R*Y.shape[0]])
                loss_list.append([R * MASK.sum()])
                mae_list.append([MAE(Y, YHAT, MASK) * MASK.sum()])
                count += MASK.sum()
        print(f"inference_time: {time.time() - inf_start}")
        print(f"avg_epoch_time: {np.mean(epoch_times)}")
        test_loss = torch.sum(torch.Tensor(loss_list).to(DEVICE) / count)
        print(f"best_val_loss: {best_val_loss.item()},  test_loss: {test_loss.item()}")
        test_mae = torch.sum(torch.Tensor(mae_list).to(DEVICE) / count)
        print(f"test_mae: {test_mae.item()}")
        print()
        break
