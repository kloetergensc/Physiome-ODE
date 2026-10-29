"""Importable training function for TIDES on Physiome-ODE.

Used by experiments/training/train_tides.py (run_final_folds.py upstream).
From physiome_ode/tides_train_fn.py of https://github.com/TaylanSoydan/TIDES
(commit 400aa0c, MIT License, see LICENSE); only the two package imports below
were changed to relative imports.

Functions
---------
_mse
    Jit-compiled masked mean squared error.
_mae
    Jit-compiled masked mean absolute error.
_build_ssm_optimizer
    Build a three-group AdamW optimizer for SSM models.
train_tides
    Train TIDES on one dataset/fold and return metrics.
"""


#
#                                                                       Modules
# =============================================================================
# Standard
import os
import random
import time
import warnings
from typing import Optional
# Third-party
import numpy as np
import torch
from torch import Tensor, jit
from torch.optim import AdamW
from tqdm import tqdm
# Local
from .tides_collate import tides_collate
from .tides_forecasting import TIDESForecastingModel
from utils import IMTS_dataset, get_data_loaders  # noqa: F401
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Taylan Soydan'
__credits__ = ['Taylan Soydan']
__status__ = 'Development'
# =============================================================================
#
# =============================================================================


# =============================================================================
@jit.script
def _mse(y: Tensor, yhat: Tensor, mask: Tensor) -> Tensor:
    return torch.mean((y[mask] - yhat[mask]) ** 2)


@jit.script
def _mae(y: Tensor, yhat: Tensor, mask: Tensor) -> Tensor:
    return torch.mean(torch.abs(y[mask] - yhat[mask]))


# =============================================================================
def _build_ssm_optimizer(model, ssm_lr, lr_factor, weight_decay):
    """Build a three-group AdamW optimizer for SSM models.

    Parameters
    ----------
    model : torch.nn.Module
        Model whose parameters will be grouped.
    ssm_lr : float
        Learning rate applied to the SSM leaf parameters.
    lr_factor : int
        Multiplier applied to ``ssm_lr`` for non-SSM parameters.
    weight_decay : float
        Weight decay applied to the regular weight group.

    Returns
    -------
    optimizer : torch.optim.AdamW
        Optimizer with three parameter groups:
        - 'ssm': Lambda_re, Lambda_im, B, C on TIDESSSM -> ssm_lr, no WD.
        - 'regular_wd': kernel weights -> ssm_lr*lr_factor, WD.
        - 'regular_no_wd': biases, log_step -> ssm_lr*lr_factor, no WD.
    """
    SSM_LEAF_NAMES = frozenset({'Lambda_re', 'Lambda_im', 'B', 'C'})
    PROJ_NAMES = frozenset({'lambda_proj', 'bc_proj'})
    NO_WD_NAMES = frozenset({'bias', 'log_step'})

    ssm_params, regular_wd_params, regular_no_wd_params = [], [], []
    for name, param in model.named_parameters():
        parts = name.split('.')
        leaf = parts[-1]
        in_proj = any(p in parts for p in PROJ_NAMES)

        if not in_proj and leaf in SSM_LEAF_NAMES:
            ssm_params.append(param)
        elif leaf in NO_WD_NAMES:
            regular_no_wd_params.append(param)
        else:
            regular_wd_params.append(param)

    regular_lr = ssm_lr * lr_factor
    return AdamW([
        {'params': ssm_params,
         'lr': ssm_lr,
         'weight_decay': 0.0},
        {'params': regular_wd_params,
         'lr': regular_lr,
         'weight_decay': weight_decay},
        {'params': regular_no_wd_params,
         'lr': regular_lr,
         'weight_decay': 0.0},
    ])
# =============================================================================


# =============================================================================
def train_tides(
        dataset,
        fold,
        data_base_path,
        # model
        hidden_size=32,
        ssm_size=16,
        ssm_blocks=4,
        num_blocks=4,
        encoder_depth=0,
        lambda_re_mode='input_dependent',
        lambda_im_mode='lti',
        bc_mode='input_dependent',
        bidir=False,
        lambda_encoder_depth=0,
        bc_rank=8,
        ff_mult=1.0,
        learn_lambda='standard',
        discretization='zoh',
        drop_rate=0.0,
        dt_min=0.001,
        dt_max=0.1,
        clip_eigs=False,
        proj_init_method='zeros',
        proj_norm='rmsnorm',
        # training
        lr=0.001,
        weight_decay=0.001,
        lr_factor=1,
        warmup_epochs=0,
        batch_size=32,
        epochs=30,
        early_stop_patience=5,
        seed=0,
        saved_models_dir='saved_models',
        verbose=True,
        wandb_run=None):
    """Train TIDES on one dataset/fold and return metrics.

    Parameters
    ----------
    dataset : str
        Dataset name.
    fold : int
        Fold index.
    data_base_path : str
        Base path to the dataset directory.
    hidden_size : int, default=32
        Hidden size of the model.
    ssm_size : int, default=16
        SSM state size.
    ssm_blocks : int, default=4
        Number of SSM blocks.
    num_blocks : int, default=4
        Number of model blocks.
    encoder_depth : int, default=0
        Depth of the encoder.
    lambda_re_mode : str, default='input_dependent'
        Mode for the real part of Lambda.
    lambda_im_mode : str, default='lti'
        Mode for the imaginary part of Lambda.
    bc_mode : str, default='input_dependent'
        Mode for B and C.
    bidir : bool, default=False
        Whether to use a bidirectional model.
    lambda_encoder_depth : int, default=0
        Depth of the Lambda encoder.
    bc_rank : int, default=8
        Rank used by B and C.
    ff_mult : float, default=1.0
        Feedforward multiplier.
    learn_lambda : str, default='standard'
        Learning mode for Lambda.
    discretization : str, default='zoh'
        Discretization method.
    drop_rate : float, default=0.0
        Dropout rate.
    dt_min : float, default=0.001
        Minimum dt.
    dt_max : float, default=0.1
        Maximum dt.
    clip_eigs : bool, default=False
        Whether to clip eigenvalues.
    proj_init_method : str, default='zeros'
        Projection initialization method.
    proj_norm : {'rmsnorm', None}, default='rmsnorm'
        RMSNorm on the projector outputs, or None for none.
    lr : float, default=0.001
        Learning rate.
    weight_decay : float, default=0.001
        Weight decay.
    lr_factor : int, default=1
        Multiplier applied to the SSM learning rate.
    warmup_epochs : int, default=0
        Number of warmup epochs.
    batch_size : int, default=32
        Mini-batch size.
    epochs : int, default=30
        Number of training epochs.
    early_stop_patience : int, default=5
        Number of epochs without improvement before early stopping.
    seed : int, default=0
        Random seed.
    saved_models_dir : str, default='saved_models'
        Directory where checkpoints are saved.
    verbose : bool, default=True
        If True, print training progress.
    wandb_run : {wandb.sdk.wandb_run.Run, None}, default=None
        Optional active wandb run. If provided, logs val_loss per epoch
        and final test metrics.

    Returns
    -------
    metrics : dict
        Dictionary with keys: val_loss, test_loss, test_mae, num_params,
        epoch_times, stopped_early.
    """
    warnings.filterwarnings(
        action='ignore', category=UserWarning, module='torch')

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(saved_models_dir, exist_ok=True)
    # Use PID so concurrent workers never share a checkpoint file
    model_path = os.path.join(
        saved_models_dir,
        f'TIDES_{dataset}_f{fold}_pid{os.getpid()}.h5')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Data
    path = os.path.join(data_base_path, dataset)
    train_loader, valid_loader, test_loader = get_data_loaders(
        fold=fold,
        path=path,
        batch_size=batch_size,
        collate_fn=tides_collate,
    )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Model
    first_batch = next(iter(train_loader))
    d_input = first_batch.values.shape[-1]

    model_config = dict(
        d_input=d_input,
        d_hidden=hidden_size,
        ssm_size=ssm_size,
        ssm_blocks=ssm_blocks,
        num_blocks=num_blocks,
        bidir=bidir,
        encoder_depth=encoder_depth,
        lambda_re_mode=lambda_re_mode,
        lambda_im_mode=lambda_im_mode,
        bc_mode=bc_mode,
        lambda_encoder_depth=lambda_encoder_depth,
        bc_rank=bc_rank,
        ff_mult=ff_mult,
        learn_lambda=learn_lambda,
        discretization=discretization,
        drop_rate=drop_rate,
        dt_min=dt_min,
        dt_max=dt_max,
        clip_eigs=clip_eigs,
        proj_init_method=proj_init_method,
        proj_norm=proj_norm,
    )
    model = TIDESForecastingModel(**model_config).to(device)
    num_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)

    if verbose:
        print(f'  d_input={d_input}  params={num_params:,}')

    if wandb_run is not None:
        wandb_run.summary['num_params'] = num_params

    # Sanity check
    with torch.no_grad():
        values, step_scale, target_mask, y_vals = (
            t.to(device) for t in first_batch)
        yhat = model(values, step_scale)
        loss = _mse(y_vals, yhat, target_mask)
        if not torch.isfinite(loss):
            raise ValueError('NaN on first forward pass')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Optimizer
    optimizer = _build_ssm_optimizer(
        model, ssm_lr=lr, lr_factor=lr_factor,
        weight_decay=weight_decay)

    # Cosine LR schedule with optional linear warmup
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=0
    )
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0 / warmup_epochs,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine],
            milestones=[warmup_epochs]
        )
    else:
        scheduler = cosine
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Training loop
    best_val_loss = float('inf')
    early_stop_count = 0
    epoch_times = []
    stopped_early = False

    pbar = tqdm(
        range(1, epochs + 1),
        desc='Training',
        mininterval=1800,
        maxinterval=1800,
    )
    for epoch in pbar:
        model.train()
        t0 = time.time()
        train_sum = 0.0
        train_count = 0
        for batch in train_loader:
            optimizer.zero_grad()
            values, step_scale, target_mask, y_vals = (
                t.to(device) for t in batch)
            yhat = model(values, step_scale)
            loss = _mse(y_vals, yhat, target_mask)
            if not torch.isfinite(loss):
                raise ValueError(f'NaN loss at epoch {epoch}')
            loss.backward()
            optimizer.step()
            n = int(target_mask.sum().item())
            train_sum += loss.item() * n
            train_count += n

        epoch_times.append(time.time() - t0)
        train_loss = (train_sum / train_count
                      if train_count > 0 else float('inf'))

        model.eval()
        val_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in valid_loader:
                values, step_scale, target_mask, y_vals = (
                    t.to(device) for t in batch)
                yhat = model(values, step_scale)
                r = _mse(y_vals, yhat, target_mask)
                n = int(target_mask.sum().item())
                val_sum += r.item() * n
                val_count += n

        val_loss = (val_sum / val_count
                    if val_count > 0 else float('inf'))

        if verbose:
            print(f'  ep {epoch:3d}  train={train_loss:.4f}  '
                  f'val={val_loss:.4f}  t={epoch_times[-1]:.1f}s',
                  flush=True)

        if wandb_run is not None:
            wandb_run.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
            })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {'state_dict': model.state_dict(),
                 'config': model_config},
                model_path)
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= early_stop_patience:
                if verbose:
                    pbar.write(f'  Early stop at epoch {epoch}')
                stopped_early = True
                break

        scheduler.step()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Test
    chp = torch.load(model_path, weights_only=False)
    model.load_state_dict(chp['state_dict'])
    model.eval()

    test_sum = 0.0
    mae_sum = 0.0
    test_count = 0
    with torch.no_grad():
        for batch in test_loader:
            values, step_scale, target_mask, y_vals = (
                t.to(device) for t in batch)
            yhat = model(values, step_scale)
            n = int(target_mask.sum().item())
            test_sum += _mse(y_vals, yhat, target_mask).item() * n
            mae_sum += _mae(y_vals, yhat, target_mask).item() * n
            test_count += n

    test_loss = (test_sum / test_count
                 if test_count > 0 else float('inf'))
    test_mae = (mae_sum / test_count
                if test_count > 0 else float('inf'))

    if wandb_run is not None:
        wandb_run.summary['best_val_loss'] = best_val_loss
        wandb_run.summary['test_loss'] = test_loss
        wandb_run.summary['test_mae'] = test_mae

    # Clean up checkpoint to save disk space
    try:
        os.remove(model_path)
    except OSError:
        pass

    return {
        'val_loss': best_val_loss,
        'test_loss': test_loss,
        'test_mae': test_mae,
        'num_params': num_params,
        'epoch_times': epoch_times,
        'stopped_early': stopped_early,
    }
# =============================================================================