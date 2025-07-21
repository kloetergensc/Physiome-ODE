import argparse
import sys
from random import SystemRandom

from utils import IMTS_dataset, get_data_loaders

# fmt: off
parser = argparse.ArgumentParser(description="Training Script for MIMIC dataset.")
parser.add_argument("-q",  "--quiet",        default=False,  const=True, help="kernel-inititialization", nargs="?")
parser.add_argument("-r",  "--run_id",       default=None,   type=str,   help="run_id")
parser.add_argument("-c",  "--config",       default=None,   type=str,   help="load external config", nargs=2)
parser.add_argument("-e",  "--epochs",       default=100,    type=int,   help="maximum epochs")
parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
parser.add_argument("-f",  "--fold",         default=4,      type=int,   help="fold number")
parser.add_argument("-bs", "--batch-size",   default=32,     type=int,   help="batch-size")
parser.add_argument("-lr", "--learn-rate",   default=0.001,  type=float, help="learn-rate")
parser.add_argument("-b",  "--betas", default=(0.9, 0.999),  type=float, help="adam betas", nargs=2)
parser.add_argument("-wd", "--weight-decay", default=0.001,  type=float, help="weight-decay")
parser.add_argument("-hs", "--hidden-size",  default=128,    type=int,   help="hidden-size")
parser.add_argument("-ls", "--latent-size",  default=128,    type=int,   help="latent-size")
parser.add_argument("-ki", "--kernel-init",  default="skew-symmetric",   help="kernel-inititialization")
parser.add_argument("-n",  "--note",         default="",     type=str,   help="Note that can be added")
parser.add_argument("-fh", "--forc-time", default=0.5, type=float, help="forecast horizon [0,1]")
parser.add_argument("-ot", "--observation-time", default=0.5, type=float, help="conditioning range [0,1]")
parser.add_argument("-nf", "--nfolds", default=5, type=int, help="#folds for crossvalidation")
parser.add_argument("-dset", "--dataset", required=True, type=str, help="Name of the dataset")

# fmt: on

ARGS = parser.parse_args()
print(" ".join(sys.argv))
experiment_id = int(SystemRandom().random() * 10000000)
print(ARGS, experiment_id)

model_path = f"saved_models/LinODE_{ARGS.dataset}_{experiment_id}.h5"

import yaml

if ARGS.config is not None:
    cfg_file, cfg_id = ARGS.config
    with open(cfg_file, "r") as file:
        cfg_dict = yaml.safe_load(file)
        vars(ARGS).update(**cfg_dict[int(cfg_id)])

import os
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, jit

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# torch.jit.enable_onednn_fusion(True)
torch.backends.cudnn.benchmark = True
# torch.multiprocessing.set_start_method('spawn')

warnings.filterwarnings(action="ignore", category=UserWarning, module="torch")


OPTIMIZER_CONFIG = {
    "lr": ARGS.learn_rate,
    "betas": ARGS.betas,
    "weight_decay": ARGS.weight_decay,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
ARGS.DEVICE = DEVICE

from models.linodenet.utils.data_utils import linodenet_collate as task_collate_fn

TRAIN_LOADER, VALID_LOADER, TEST_LOADER = get_data_loaders(
    fold=ARGS.fold,
    path=f"../../data/final/{ARGS.dataset}/",
    batch_size=ARGS.batch_size,
    collate_fn=task_collate_fn,
)
EVAL_LOADERS = {"train": TRAIN_LOADER, "valid": VALID_LOADER, "test": TEST_LOADER}


def MSE(y: Tensor, yhat: Tensor) -> Tensor:
    return torch.mean((y - yhat) ** 2)


def MAE(y: Tensor, yhat: Tensor) -> Tensor:
    return torch.mean(torch.abs(y - yhat))


def RMSE(y: Tensor, yhat: Tensor) -> Tensor:
    return torch.sqrt(torch.mean((y - yhat) ** 2))


METRICS = {
    "RMSE": jit.script(RMSE),
    "MSE": jit.script(MSE),
    "MAE": jit.script(MAE),
}
LOSS = jit.script(MSE)


from models.linodenet.models import LinODEnet, ResNet, embeddings, filters, system

batch = next(iter(TRAIN_LOADER))
T, X, M, TY, Y, MY = (tensor for tensor in batch)

MODEL_CONFIG = {
    "__name__": "LinODEnet",
    "input_size": X.shape[-1],
    "hidden_size": ARGS.hidden_size,
    "latent_size": ARGS.latent_size,
    "Filter": filters.SequentialFilter.HP | {"autoregressive": True},
    "System": system.LinODECell.HP | {"kernel_initialization": ARGS.kernel_init},
    "Encoder": ResNet.HP,
    "Decoder": ResNet.HP,
    "Embedding": embeddings.ConcatEmbedding.HP,
    "Projection": embeddings.ConcatProjection.HP,
}

MODEL = LinODEnet(**MODEL_CONFIG).to(DEVICE)
MODEL = torch.jit.script(MODEL)


def predict_fn(model, batch) -> tuple[Tensor, Tensor]:
    """Get targets and predictions."""
    T, X, M, _, Y, MY = (tensor.to(DEVICE) for tensor in batch)
    YHAT = model(T, X)
    return Y[MY], YHAT[M]


batch = next(iter(TRAIN_LOADER))
MODEL.zero_grad(set_to_none=True)

# Forward
Y, YHAT = predict_fn(MODEL, batch)

# Backward
R = LOSS(Y, YHAT)
assert torch.isfinite(R).item(), "Model Collapsed!"
R.backward()

# Reset
MODEL.zero_grad(set_to_none=True)

from torch.optim import AdamW

OPTIMIZER = AdamW(MODEL.parameters(), **OPTIMIZER_CONFIG)


def eval_model(MODEL, DATA_LOADER, ARGS):
    loss_list = []
    mae_list = []
    MODEL.eval()
    with torch.no_grad():
        count = 0
        for i, batch in enumerate(DATA_LOADER):
            Y, YHAT = predict_fn(MODEL, batch)
            R = LOSS(Y, YHAT) * len(Y)
            count += len(Y)
            loss_list.append([R])
            mae_list.append([MAE(Y, YHAT) * len(Y)])
    val_loss = torch.sum(torch.Tensor(loss_list).to(ARGS.DEVICE) / count)
    vae_mae = torch.sum(torch.Tensor(mae_list).to(ARGS.DEVICE) / count)
    return val_loss, vae_mae


total_num_batches = 0
print("Start Training")
ovr_start_time = time.time()
# print(MODEL)
early_stop = 0
best_val_loss = 1e8
epoch_times = []

for epoch in range(ARGS.epochs):
    loss_list = []
    start_time = time.time()
    for i, batch in enumerate(TRAIN_LOADER):
        MODEL.zero_grad(set_to_none=True)

        # Forward
        Y, YHAT = predict_fn(MODEL, batch)
        R = LOSS(Y, YHAT)
        assert torch.isfinite(R).item(), "Model Collapsed!"
        loss_list.append([R])
        # Backward
        R.backward()
        OPTIMIZER.step()
    epoch_time = time.time()
    train_loss = torch.mean(torch.Tensor(loss_list))
    count = 0
    # After each epoch compute validation error
    val_loss, vae_mae = eval_model(MODEL, VALID_LOADER, ARGS)
    epoch_times.append(epoch_time - start_time)
    print(
        f"{epoch:3.0f} Train:{train_loss.item():4.4f}  VAL: {val_loss.item():4.4f}   EPOCH_TIME: {epoch_times[-1]:3.2f}"
    )

    # if current val_loss is less than the best val_loss save the parameters
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # best_model = deepcopy(model.state_dict())
        torch.save(
            {
                "ARGS": ARGS,
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
    # Compute test_loss if all the epochs or completed or early stop if val_loss did not improve for # many appochs
    if (early_stop == ARGS.patience) or (epoch == ARGS.epochs - 1):
        print(f"tot_train_time: {time.time()-ovr_start_time}")
        if early_stop == ARGS.patience:
            print(
                f"Early stopping because of no improvement in val. metric for {ARGS.patience} epochs"
            )
        else:
            print("Exhausted all the epochs")
        chp = torch.load(model_path, weights_only=False)
        MODEL.load_state_dict(chp["state_dict"])
        start_inf = time.time()
        test_loss, test_mae = eval_model(MODEL, TEST_LOADER, ARGS)
        print(f"inference_time: {time.time()-start_inf}")
        print(f"best_val_loss: {best_val_loss.item()}, test_loss: {test_loss.item()}")
        print(f"test_mae: {test_mae.item()}")
        print(f"avg_epoch_time: {np.mean(epoch_times)}")
        num_params = sum(p.numel() for p in MODEL.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {num_params}")
        break
