import argparse
import logging
import os
import sys
import time
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from random import SystemRandom

import numpy as np
import torch
from models.neuralflow import data_utils
from torch import Tensor, jit
from utils import IMTS_dataset, get_data_loaders

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

parser = argparse.ArgumentParser("Neural flows")
# Training loop ARGS
parser.add_argument("--epochs", type=int, default=1000, help="Max training epochs")
parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument(
    "--weight-decay", type=float, default=0, help="Weight decay (regularization)"
)
parser.add_argument(
    "--lr-scheduler-step",
    type=int,
    default=20,
    help="Every how many steps to perform lr decay",
)
parser.add_argument(
    "--lr-decay", type=float, default=0.33, help="Multiplicative lr decay factor"
)
parser.add_argument(
    "-b", "--betas", default=(0.9, 0.999), type=float, help="adam betas", nargs=2
)
parser.add_argument("--batch-size", type=int, default=50)
parser.add_argument("--clip", type=float, default=1, help="Gradient clipping")
parser.add_argument("-f", "--fold", default=0, type=int, help="fold number")

# NN ARGS
parser.add_argument(
    "--hidden-layers", type=int, default=1, help="Number of hidden layers"
)
parser.add_argument("--hidden-dim", type=int, default=1, help="Size of hidden layer")
parser.add_argument(
    "--activation", type=str, default="Tanh", help="Hidden layer activation"
)
parser.add_argument(
    "--final-activation", type=str, default="Identity", help="Last layer activation"
)

# Flow model ARGS
parser.add_argument(
    "--flow-model",
    type=str,
    default="resnet",
    help="Model name",
    choices=["coupling", "resnet", "gru"],
)
parser.add_argument("--flow-layers", type=int, default=1, help="Number of flow layers")
parser.add_argument(
    "--time-net",
    type=str,
    default="TimeTanh",
    help="Name of time net",
    choices=["TimeFourier", "TimeFourierBounded", "TimeLinear", "TimeTanh"],
)
parser.add_argument(
    "--time-hidden-dim",
    type=int,
    default=1,
    help="Number of time features (only for Fourier)",
)

parser.add_argument(
    "--odenet",
    type=str,
    default="concat",
    help="Type of ODE network",
    choices=["concat", "gru"],
)  # gru only in GOB
parser.add_argument(
    "--solver",
    type=str,
    default="dopri5",
    help="ODE solver",
    choices=["dopri5", "rk4", "euler"],
)
parser.add_argument("--solver_step", type=float, default=0.05, help="Fixed solver step")
parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance")
parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance")

parser.add_argument(
    "--mixing", type=float, default=0.0001, help="Ratio between KL and update loss"
)
parser.add_argument(
    "--gob_prep_hidden",
    type=int,
    default=10,
    help="Size of hidden state for covariates",
)
parser.add_argument(
    "--gob_cov_hidden", type=int, default=50, help="Size of hidden state for covariates"
)
parser.add_argument(
    "--gob_p_hidden",
    type=int,
    default=25,
    help="Size of hidden state for initialization",
)
parser.add_argument(
    "--invertible", type=int, default=1, help="If network is invertible", choices=[0, 1]
)
parser.add_argument(
    "-fh", "--forc-time", default=0.5, type=float, help="forecast horizon [0,1]"
)
parser.add_argument(
    "-ot",
    "--observation-time",
    default=0.5,
    type=float,
    help="conditioning range [0,1]",
)
parser.add_argument(
    "-nf", "--nfolds", default=5, type=int, help="#folds for crossvalidation"
)
parser.add_argument(
    "-dset", "--dataset", required=True, type=str, help="Name of the dataset"
)

ARGS = parser.parse_args()

print(" ".join(sys.argv))
experiment_id = int(SystemRandom().random() * 10000000)
print(ARGS, experiment_id)
ARGS.data = ARGS.dataset

model_path = f"saved_models/LinODE_{ARGS.dataset}_{experiment_id}.h5"

warnings.filterwarnings(action="ignore", category=UserWarning, module="torch")
logging.basicConfig(level=logging.WARN)


# Configure the optimizer
OPTIMIZER_CONFIG = {
    "lr": ARGS.lr,
    "betas": torch.tensor(ARGS.betas),
    "weight_decay": ARGS.weight_decay,
}

# get the datasets from tsdm
# configure dataloaders. We need different configurations for train and evaluation

TRAIN_LOADER, VALID_LOADER, TEST_LOADER = get_data_loaders(
    fold=ARGS.fold,
    path=f"../../data/final/{ARGS.dataset}/",
    batch_size=ARGS.batch_size,
    collate_fn=data_utils.tsdm_collate_val,
)
EVAL_LOADERS = {"train": TRAIN_LOADER, "valid": VALID_LOADER, "test": TEST_LOADER}


def MSE(y: Tensor, yhat: Tensor, mask: Tensor) -> Tensor:
    err = torch.mean((y[mask] - yhat[mask]) ** 2)
    return err


def MAE(y: Tensor, yhat: Tensor, mask: Tensor) -> Tensor:
    err = torch.mean(torch.abs(y[mask] - yhat[mask]))
    return err


def RMSE(y: Tensor, yhat: Tensor, mask: Tensor) -> Tensor:
    err = torch.sqrt(torch.sum(mask * (y - yhat) ** 2, 1) / (torch.sum(mask, 1)))
    return torch.mean(err)


METRICS = {
    "RMSE": jit.script(RMSE),
    "MSE": jit.script(MSE),
    "MAE": jit.script(MAE),
}
LOSS = jit.script(MSE)


def eval_model(model, dataloader, ARGS) -> Tensor:
    loss_list = []
    mae_list = []
    model.eval()
    with torch.no_grad():
        count = 0
        for i, b in enumerate(dataloader):
            assert b["X_val"] is not None
            hT, _, _, _, _, p_vec = model(
                b["times"],
                b["num_obs"],
                b["X"].to(DEVICE),
                b["M"].to(DEVICE),
                delta_t=ARGS.solver_step,
                cov=b["cov"].to(DEVICE),
                return_path=True,
                val_times=b["times_val"],
            )
            m, v = torch.chunk(p_vec, 2, dim=1)

            z_reord, mask_reord = [], []
            val_numobs = torch.Tensor([len(x) for x in b["times_val"]])
            for ind in range(0, int(torch.max(val_numobs).item())):
                idx = val_numobs > ind
                zero_tens = torch.Tensor([0])
                z_reord.append(
                    b["X_val"][
                        (
                            torch.cat((zero_tens, torch.cumsum(val_numobs, dim=0)))[
                                :-1
                            ][idx]
                            + ind
                        ).long()
                    ]
                )
                mask_reord.append(
                    b["M_val"][
                        (
                            torch.cat((zero_tens, torch.cumsum(val_numobs, dim=0)))[
                                :-1
                            ][idx]
                            + ind
                        ).long()
                    ]
                )

            Y = torch.cat(z_reord).to(DEVICE)
            MASK = torch.cat(mask_reord).to(DEVICE)
            YHAT = m
            R = LOSS(Y, YHAT, MASK)
            loss_list.append([R * MASK.sum()])
            mae_list.append([MAE(Y, YHAT, MASK) * MASK.sum()])
            count += MASK.sum()
        val_loss = torch.sum(torch.Tensor(loss_list).to(DEVICE) / count)
        mae_loss = torch.sum(torch.Tensor(mae_list).to(DEVICE) / count)

    return val_loss, mae_loss


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARGS.device = DEVICE

b = next(iter(TRAIN_LOADER))
input_size = b["M"].shape[-1]

from models.neuralflow.nnfowithbayesianjumps import NNFOwithBayesianJumps

model = NNFOwithBayesianJumps(
    input_size=input_size,
    args=ARGS,
    cov_hidden=ARGS.gob_cov_hidden,
    cov_size=1,
    mixing=ARGS.mixing,
    p_hidden=ARGS.gob_p_hidden,
    prep_hidden=ARGS.gob_prep_hidden,
)
model.to(DEVICE)

OPTIMIZER = torch.optim.Adam(
    model.parameters(), lr=ARGS.lr, weight_decay=ARGS.weight_decay
)
class_criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
print("Start Training")
ovr_train_start = time.time()
E_times = []
# print(model)
early_stop = 0
best_val_loss = 1e8
for epoch in range(ARGS.epochs):
    start = time.time()
    total_train_loss = 0
    auc_total_train = 0
    tot_loglik_loss = 0
    loss_list = []
    model.train()
    for i, b in enumerate(TRAIN_LOADER):
        OPTIMIZER.zero_grad()
        _, loss, _, _, _ = model(
            b["times"],
            b["num_obs"],
            b["X"].to(DEVICE),
            b["M"].to(DEVICE),
            delta_t=ARGS.solver_step,
            cov=b["cov"].to(DEVICE),
            val_times=b["times_val"],
        )
        R = loss / b["y"].size(0)
        assert torch.isfinite(R).item(), "Model Collapsed!"
        loss_list.append([R])
        R.backward()
        OPTIMIZER.step()
    e_time = time.time() - start
    E_times.append(e_time)
    train_loss = torch.mean(torch.Tensor(loss_list))
    count = 0
    loss_list = []

    # After each epoch compute validation error
    val_loss, mae_loss = eval_model(model, VALID_LOADER, ARGS)
    print(
        f"{epoch:3.0f} Train:{train_loss.item():4.4f}  VAL: {val_loss.item():4.4f}    EPOCH_TIME: {e_time:3.2f}"
    )
    # if current val_loss is less than the best val_loss save the parameters
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # best_model = deepcopy(model.state_dict())
        torch.save(
            {
                "ARGS": ARGS,
                "epoch": epoch,
                "state_dict": deepcopy(model.state_dict()),
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
        print(f"tot_train_time: {time.time() - ovr_train_start}")
        if early_stop == ARGS.patience:
            print(
                f"Early stopping because of no improvement in val. metric for {ARGS.patience} epochs"
            )
        else:
            print("Exhausted all the epochs")
        chp = torch.load(model_path, weights_only=False)
        model.load_state_dict(chp["state_dict"])
        inf_start = time.time()
        test_loss, test_mae = eval_model(model, TEST_LOADER, ARGS)
        print(f"inference_time: {time.time() - inf_start}")
        print(f"best_val_loss: {best_val_loss.item()}, test_loss: {test_loss.item()}")
        print(f"test_mae: {test_mae.item()}")
        print(f"avg_epoch_time: {np.mean(E_times)}")
        break
