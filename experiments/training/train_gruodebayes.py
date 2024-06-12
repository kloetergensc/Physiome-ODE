import argparse
import sys
from random import SystemRandom

# fmt: off
parser = argparse.ArgumentParser(description="Training Script for USHCN dataset.")
parser.add_argument("-q",  "--quiet",        default=False,  const=True, help="kernel-inititialization", nargs="?")
parser.add_argument("-r",  "--run_id",       default=None,   type=str,   help="run_id")
parser.add_argument("-c",  "--config",       default=None,   type=str,   help="load external config", nargs=2)
parser.add_argument("-e",  "--epochs",       default=100,    type=int,   help="maximum epochs")
parser.add_argument("-f",  "--fold",         default=0,      type=int,   help="fold number")
parser.add_argument("-bs", "--batch-size",   default=64,     type=int,   help="batch-size")
parser.add_argument("-lr", "--learn-rate",   default=0.001,  type=float, help="learn-rate")
parser.add_argument("-b",  "--betas", default=(0.9, 0.999),  type=float, help="adam betas", nargs=2)
parser.add_argument("-wd", "--weight-decay", default=0.001,  type=float, help="weight-decay")
parser.add_argument("-ki", "--kernel-init",  default="skew-symmetric",   help="kernel-inititialization")
parser.add_argument("-dr",  "--dropout", default=0.2,   type=float,   help="")
parser.add_argument("-hs",  "--hidden-size", default=50,   type=int,   help="")

parser.add_argument("-es", "--early-stop",default=10)
parser.add_argument("-so", "--solver", default="euler", choices = ["euler", "dopri5"])

parser.add_argument("-nf", "--nfolds", default=5, type=int, help="#folds for crossvalidation")
parser.add_argument("-fh", "--forc-time", default=0.5, type=float, help="forecast horizon [0,1]")
parser.add_argument("-ot", "--observation-time", default=0.5, type=float, help="conditioning range [0,1]")
parser.add_argument("-bn", "--benchmark", default="imts_bench")
parser.add_argument("-dset", "--dataset", required=True, type=str, help="Name of the dataset")


import time

# fmt: on

ARGS = parser.parse_args()
print(" ".join(sys.argv))
experiment_id = int(SystemRandom().random() * 10000000)
print(ARGS, experiment_id)
model_path = f"experiments/training/saved_models/GOB_{ARGS.dataset}_{experiment_id}.h5"


params_dict = {}
params_dict["hidden_size"] = ARGS.hidden_size
params_dict["p_hidden"] = 25
params_dict["prep_hidden"] = 10
params_dict["logvar"] = True
params_dict["mixing"] = 1e-4  # Weighting between KL loss and MSE loss.
params_dict["delta_t"] = 0.0006
params_dict["T"] = 1
params_dict["lambda"] = 0  # Weighting between classification and MSE loss.

params_dict["classification_hidden"] = 2
params_dict["cov_hidden"] = 50
params_dict["weight_decay"] = ARGS.weight_decay
params_dict["dropout_rate"] = ARGS.dropout
params_dict["lr"] = 0.001
params_dict["full_gru_ode"] = True
params_dict["no_cov"] = True
params_dict["impute"] = False
params_dict["verbose"] = 0  # from 0 to 3 (highest)

import yaml

if ARGS.config is not None:
    cfg_file, cfg_id = ARGS.config
    with open(cfg_file, "r") as file:
        cfg_dict = yaml.safe_load(file)
        vars(ARGS).update(**cfg_dict[int(cfg_id)])

print(ARGS)


import logging
import os
import pdb
import random
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from IPython.core.display import HTML
from torch import Tensor, jit
from tqdm.autonotebook import tqdm, trange
from utils import IMTS_dataset, get_data_loaders

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# torch.jit.enable_onednn_fusion(True)
torch.backends.cudnn.benchmark = True
# torch.multiprocessing.set_start_method('spawn')

warnings.filterwarnings(action="ignore", category=UserWarning, module="torch")
logging.basicConfig(level=logging.WARN)
HTML("<style>.jp-OutputArea-prompt:empty {padding: 0; border: 0;}</style>")

import time

import models.gruodebayes.data_utils as data_utils
import models.gruodebayes.models as gru_ode_bayes
import tqdm
from sklearn.metrics import roc_auc_score

# ## Hyperparameter choices

# In[55]:


OPTIMIZER_CONFIG = {
    "lr": ARGS.learn_rate,
    "betas": torch.tensor(ARGS.betas),
    "weight_decay": ARGS.weight_decay,
}
# In[57]:


TRAIN_LOADER, VALID_LOADER, TEST_LOADER = get_data_loaders(
    fold=ARGS.fold,
    path=f"data/IMTS_benchmark_datasets/{ARGS.benchmark}/{ARGS.dataset}/",
    observation_time=ARGS.observation_time,
    forecasting_horizon=ARGS.forc_time,
    batch_size=ARGS.batch_size,
    collate_fn=data_utils.tsdm_collate_val,
)
EVAL_LOADERS = {"train": TRAIN_LOADER, "valid": VALID_LOADER, "test": TEST_LOADER}


# ## Initialize DataLoaders
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


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


b = next(iter(TRAIN_LOADER))

params_dict["input_size"] = b["M"].shape[-1]
params_dict["cov_size"] = b["cov"].shape[-1]


nnfwobj = gru_ode_bayes.NNFOwithBayesianJumps(
    input_size=params_dict["input_size"],
    hidden_size=params_dict["hidden_size"],
    p_hidden=params_dict["p_hidden"],
    prep_hidden=params_dict["prep_hidden"],
    logvar=params_dict["logvar"],
    mixing=params_dict["mixing"],
    classification_hidden=params_dict["classification_hidden"],
    cov_size=params_dict["cov_size"],
    cov_hidden=params_dict["cov_hidden"],
    dropout_rate=params_dict["dropout_rate"],
    full_gru_ode=params_dict["full_gru_ode"],
    impute=params_dict["impute"],
    solver=ARGS.solver,
)
nnfwobj.to(DEVICE)

optimizer = torch.optim.Adam(
    nnfwobj.parameters(), lr=params_dict["lr"], weight_decay=params_dict["weight_decay"]
)
class_criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
print("Start Training")
val_metric_prev = 1000


def test_evaluation(model, params_dict, class_criterion, DEVICE, dl_test):
    chp = torch.load(model_path)
    model.load_state_dict(chp["state_dict"])
    with torch.no_grad():
        model.eval()
        total_loss_test = 0
        auc_total_test = 0
        loss_test = 0
        mse_test = 0
        mae_test = 0
        corr_test = 0
        num_obs = 0
        for i, b in enumerate(dl_test):
            times = b["times"]
            time_ptr = b["time_ptr"]
            X = b["X"].to(DEVICE)
            M = b["M"].to(DEVICE)
            obs_idx = b["obs_idx"]
            cov = b["cov"].to(DEVICE)
            labels = b["y"].to(DEVICE)
            batch_size = labels.size(0)

            if b["X_val"] is not None:
                X_val = b["X_val"].to(DEVICE)
                M_val = b["M_val"].to(DEVICE)
                times_val = b["times_val"]
                times_idx = b["index_val"]

            h0 = (
                0  # torch.zeros(labels.shape[0], params_dict["hidden_size"]).to(DEVICE)
            )
            hT, loss, class_pred, t_vec, p_vec, h_vec, _, _ = model(
                times,
                time_ptr,
                X,
                M,
                obs_idx,
                delta_t=params_dict["delta_t"],
                T=params_dict["T"],
                cov=cov,
                return_path=True,
            )
            total_loss = (
                loss + params_dict["lambda"] * class_criterion(class_pred, labels)
            ) / batch_size

            try:
                auc_test = roc_auc_score(labels.cpu(), torch.sigmoid(class_pred).cpu())
            except ValueError:
                if params_dict["verbose"] >= 3:
                    print("Only one class. AUC is wrong")
                auc_test = 0
                pass

            if params_dict["lambda"] == 0:
                t_vec = np.around(
                    t_vec, str(params_dict["delta_t"])[::-1].find(".")
                ).astype(
                    np.float32
                )  # Round floating points error in the time vector.
                p_val = data_utils.extract_from_path(t_vec, p_vec, times_val, times_idx)
                m, v = torch.chunk(p_val, 2, dim=1)
                last_loss = (data_utils.log_lik_gaussian(X_val, m, v) * M_val).sum()
                mse_loss = (torch.pow(X_val - m, 2) * M_val).sum()
                mae_loss = (torch.abs(X_val - m) * M_val).sum()
                corr_test_loss = data_utils.compute_corr(X_val, m, M_val)

                loss_test += last_loss.cpu().numpy()
                num_obs += M_val.sum().cpu().numpy()
                mse_test += mse_loss.cpu().numpy()
                mae_test += mae_loss.cpu().numpy()
                corr_test += corr_test_loss.cpu().numpy()
            else:
                num_obs = 1

            total_loss_test += total_loss.cpu().detach().numpy()
            auc_total_test += auc_test

        loss_test /= num_obs
        mse_test /= num_obs
        mae_test /= num_obs
        auc_total_test /= i + 1

        return (loss_test, auc_total_test, mse_test, mae_test)


early_stop = 0
ovr_train_start = time.time()
epoch_times = []
for epoch in range(ARGS.epochs):
    nnfwobj.train()
    total_train_loss = 0
    auc_total_train = 0
    tot_loglik_loss = 0
    start = time.time()
    for i, batch in enumerate(TRAIN_LOADER):
        optimizer.zero_grad()
        times = b["times"]
        time_ptr = b["time_ptr"]
        X = b["X"].to(DEVICE)
        M = b["M"].to(DEVICE)
        obs_idx = b["obs_idx"]
        cov = b["cov"].to(DEVICE)
        labels = b["y"].to(DEVICE)
        batch_size = labels.size(0)
        h0 = 0  # torch.zeros(labels.shape[0], params_dict["hidden_size"]).to(DEVICE)
        hT, loss, class_pred, mse_loss = nnfwobj(
            times,
            time_ptr,
            X,
            M,
            obs_idx,
            delta_t=params_dict["delta_t"],
            T=params_dict["T"],
            cov=cov,
        )

        total_loss = (
            loss + params_dict["lambda"] * class_criterion(class_pred, labels)
        ) / batch_size
        total_train_loss += total_loss
        tot_loglik_loss += mse_loss
        try:
            auc_total_train += roc_auc_score(
                labels.detach().cpu(), torch.sigmoid(class_pred).detach().cpu()
            )
        except ValueError:
            if params_dict["verbose"] >= 3:
                print("Single CLASS ! AUC is erroneous")
            pass

        total_loss.backward()
        optimizer.step()
    epoch_times.append(time.time() - start)

    info = {
        "training_loss": total_train_loss.detach().cpu().numpy() / (i + 1),
        "AUC_training": auc_total_train / (i + 1),
        "loglik_loss": tot_loglik_loss.detach().cpu().numpy(),
    }
    # for tag, value in info.items():
    #     logger.scalar_summary(tag, value, epoch)
    data_utils.adjust_learning_rate(optimizer, epoch, params_dict["lr"])

    with torch.no_grad():
        nnfwobj.eval()
        total_loss_val = 0
        auc_total_val = 0
        loss_val = 0
        mse_val = 0
        corr_val = 0
        num_obs = 0
        for i, b in enumerate(VALID_LOADER):
            times = b["times"]
            time_ptr = b["time_ptr"]
            X = b["X"].to(DEVICE)
            M = b["M"].to(DEVICE)
            obs_idx = b["obs_idx"]
            cov = b["cov"].to(DEVICE)
            labels = b["y"].to(DEVICE)
            batch_size = labels.size(0)
            # pdb.set_trace()
            if b["X_val"] is not None:
                X_val = b["X_val"].to(DEVICE)
                M_val = b["M_val"].to(DEVICE)
                times_val = b["times_val"]
                times_idx = b["index_val"]

            h0 = (
                0  # torch.zeros(labels.shape[0], params_dict["hidden_size"]).to(DEVICE)
            )
            hT, loss, class_pred, t_vec, p_vec, h_vec, _, _ = nnfwobj(
                times,
                time_ptr,
                X,
                M,
                obs_idx,
                delta_t=params_dict["delta_t"],
                T=params_dict["T"],
                cov=cov,
                return_path=True,
            )
            total_loss = (
                loss + params_dict["lambda"] * class_criterion(class_pred, labels)
            ) / batch_size

            try:
                auc_val = roc_auc_score(labels.cpu(), torch.sigmoid(class_pred).cpu())
            except ValueError:
                auc_val = 0.5
                if params_dict["verbose"] >= 3:
                    print("Only one class : AUC is erroneous")
                pass

            if params_dict["lambda"] == 0:
                t_vec = np.around(
                    t_vec, str(params_dict["delta_t"])[::-1].find(".")
                ).astype(
                    np.float32
                )  # Round floating points error in the time vector.
                # pdb.set_trace()
                p_val = data_utils.extract_from_path(t_vec, p_vec, times_val, times_idx)
                m, v = torch.chunk(p_val, 2, dim=1)
                last_loss = (data_utils.log_lik_gaussian(X_val, m, v) * M_val).sum()
                mse_loss = (torch.pow(X_val - m, 2) * M_val).sum()
                corr_val_loss = data_utils.compute_corr(X_val, m, M_val)

                loss_val += last_loss.cpu().numpy()
                num_obs += M_val.sum().cpu().numpy()
                mse_val += mse_loss.cpu().numpy()
                corr_val += corr_val_loss.cpu().numpy()
            else:
                num_obs = 1

            total_loss_val += total_loss.cpu().detach().numpy()
            auc_total_val += auc_val

        loss_val /= num_obs
        mse_val /= num_obs
        # info = { 'validation_loss' : total_loss_val/(i+1), 'AUC_validation' : auc_total_val/(i+1),
        #         'loglik_loss' : loss_val, 'validation_mse' : mse_val, 'correlation_mean' : np.nanmean(corr_val),
        #        'correlation_max': np.nanmax(corr_val), 'correlation_min': np.nanmin(corr_val)}
        # print(f"Total validation loss at epoch {epoch}: {total_loss_val/(i+1)}")
        # print(f"Validation AUC at epoch {epoch}: {auc_total_val/(i+1)}")
        # print(f"Validation loss (loglik) at epoch {epoch}: {loss_val:.5f}. MSE : {mse_val:.5f}. Correlation : {np.nanmean(corr_val):.5f}. Num obs = {num_obs}")

        val_metric = mse_val
        print(
            f"{epoch:3.0f} Train:{(tot_loglik_loss.detach().cpu().numpy()):4.4f}  VAL: {mse_val:4.4f}    EPOCH_TIME: {epoch_times[-1]:3.2f}"
        )
        if val_metric < val_metric_prev:
            # print(f"New highest validation metric reached ! : {val_metric}")
            # print("Saving Model")
            torch.save(
                {
                    "args": ARGS,
                    "epoch": epoch,
                    "state_dict": nnfwobj.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": total_train_loss.detach().cpu().numpy() / (i + 1),
                },
                model_path,
            )
            # torch.save(nnfwobj.state_dict(),f"trained_models/{str(experiment_id)}_MAX.pt")
            val_metric_prev = val_metric
            early_stop = 0
        else:
            early_stop += 1
        if (early_stop == ARGS.early_stop) or (epoch == ARGS.epochs - 1):
            print(f"tot_train_time: {time.time()- ovr_train_start}")
            inference_time_start = time.time()
            test_loglik, test_auc, test_mse, test_mae = test_evaluation(
                nnfwobj, params_dict, class_criterion, DEVICE, TEST_LOADER
            )
            print(f"inference_time: {time.time()- inference_time_start}")
            print(f"avg_epoch_time: {np.mean(epoch_times)}")
            # print(f"Test loglik loss at epoch {epoch} : {test_loglik}")
            # print(f"Test AUC loss at epoch {epoch} : {test_auc}")
            # print(f"Test MSE loss at epoch{epoch} : {test_mse}")
            print(f"best_val_loss: {val_metric_prev}, test_loss: {test_mse}")
            print(f"test_mae: {test_mae}")
            break
