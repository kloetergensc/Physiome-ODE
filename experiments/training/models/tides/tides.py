"""
PyTorch TIDES (Time-aware Input-Dependent State-space).

A diagonal S5-style SSM whose Re(Lambda), B and C are projected from the
input at every step, while the discretization step stays the observed time
interval: pass the time since the previous observation as `step_scale`.

Architecture faithfully follows the original JAX reference
implementation of the input-dependent S5 model.

SSM utilities (associative scan, discretization, HiPPO initialization)
adapted from the public S5 package
(https://github.com/lindermanlab/S5, Apache-2.0).

Modes, set separately for Re(Lambda), Im(Lambda) and (B, C)
- 'lti'             : static, as in S5
- 'input_dependent' : projected from the input at each step
                      (the default for Re(Lambda) and B, C)

Block structure: norm -> SSM -> GELU -> dropout -> GLU -> dropout -> residual

step_scale is a float (regular sampling), an (L,) tensor shared by the batch,
or a (B, L) tensor; it discretizes the continuous-time SSM at the observed
intervals.

Public API
- TIDES                   : encoder, (B, L, d_input) -> (B, L, d_hidden)
- TIDESClassifier         : encoder + mean-pool + linear head, -> (B, num_classes)
- step_scale_from_indices : step sizes for a sequence subsampled from a regular grid
"""

import math
import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Callable, Literal, Optional, Tuple
from torch.utils._pytree import tree_flatten, tree_unflatten

from ._hub import PyTorchModelHubMixin, hub_kwargs


# ══════════════════════════════════════════════════════════════════════════════
# 1. Associative Scan   (adapted from s5.jax_compat)
# ══════════════════════════════════════════════════════════════════════════════

def _interleave(a: torch.Tensor, b: torch.Tensor, axis: int) -> torch.Tensor:
    b_trunc = a.shape[axis] == b.shape[axis] + 1
    if b_trunc:
        pad = [0, 0] * b.ndim
        pad[(b.ndim - axis - 1) * 2 + 1] = 1
        b = F.pad(b, pad)
    stacked = torch.stack([a, b], dim=axis + 1)
    interleaved = torch.flatten(stacked, start_dim=axis, end_dim=axis + 1)
    if b_trunc:
        interleaved = torch.ops.aten.slice(
            interleaved, axis, 0, b.shape[axis] + a.shape[axis] - 1
        )
    return interleaved


def _combine(tree, operator, a_flat, b_flat):
    a = tree_unflatten(a_flat, tree)
    b = tree_unflatten(b_flat, tree)
    c = operator(a, b)
    c_flat, _ = tree_flatten(c)
    return c_flat


def _scan(tree, operator, elems, axis: int):
    num_elems = elems[0].shape[axis]
    if num_elems < 2:
        return elems
    reduced = _combine(
        tree, operator,
        [torch.ops.aten.slice(e, axis, 0, -1, 2) for e in elems],
        [torch.ops.aten.slice(e, axis, 1, None, 2) for e in elems],
    )
    odd_elems = _scan(tree, operator, reduced, axis)
    if num_elems % 2 == 0:
        even_elems = _combine(
            tree, operator,
            [torch.ops.aten.slice(e, axis, 0, -1) for e in odd_elems],
            [torch.ops.aten.slice(e, axis, 2, None, 2) for e in elems],
        )
    else:
        even_elems = _combine(
            tree, operator,
            odd_elems,
            [torch.ops.aten.slice(e, axis, 2, None, 2) for e in elems],
        )
    even_elems = [
        torch.cat([torch.ops.aten.slice(elem, axis, 0, 1), result], dim=axis)
        if result.numel() > 0 and elem.shape[axis] > 0
        else result if result.numel() > 0
        else torch.ops.aten.slice(elem, axis, 0, 1)
        for (elem, result) in zip(elems, even_elems)
    ]
    return list(map(partial(_interleave, axis=axis), even_elems, odd_elems))


def associative_scan(operator: Callable, elems, axis: int = 0, reverse: bool = False):
    """PyTorch port of jax.lax.associative_scan."""
    elems_flat, tree = tree_flatten(elems)
    if reverse:
        elems_flat = [torch.flip(e, [axis]) for e in elems_flat]
    num_elems = int(elems_flat[0].shape[axis])
    assert all(int(e.shape[axis]) == num_elems for e in elems_flat[1:])
    scans = _scan(tree, operator, elems_flat, axis)
    if reverse:
        scans = [torch.flip(s, [axis]) for s in scans]
    return tree_unflatten(scans, tree)


def _binary_operator(
    q_i: Tuple[torch.Tensor, torch.Tensor],
    q_j: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Binary operator for parallel scan of diagonal SSM recurrence."""
    A_i, Bu_i = q_i
    A_j, Bu_j = q_j
    return A_j * A_i, torch.addcmul(Bu_j, A_j, Bu_i)


# ══════════════════════════════════════════════════════════════════════════════
# 2. HiPPO Initialization   (adapted from s5.init)
# ══════════════════════════════════════════════════════════════════════════════

def _make_HiPPO(N: int) -> np.ndarray:
    P = np.sqrt(1 + 2 * np.arange(N))
    A = P[:, None] * P[None, :]
    A = np.tril(A) - np.diag(np.arange(N))
    return -A


def _make_DPLR_HiPPO(N: int):
    """DPLR HiPPO-LegS decomposition. Returns (Lambda, P_lowrank, B, V)."""
    hippo = _make_HiPPO(N)
    P = np.sqrt(np.arange(N) + 0.5)
    B = np.sqrt(2 * np.arange(N) + 1.0)
    S = hippo + P[:, None] * P[None, :]
    S_diag = np.diagonal(S)
    Lambda_real = np.mean(S_diag) * np.ones_like(S_diag)
    Lambda_imag, V = np.linalg.eigh(S * -1j)
    P = V.conj().T @ P
    B = V.conj().T @ B
    Lambda = Lambda_real + 1j * Lambda_imag
    return Lambda, P, B, V


def _init_log_steps(H: int, dt_min: float, dt_max: float) -> torch.Tensor:
    """Sample log-steps uniformly in [log(dt_min), log(dt_max)]."""
    return torch.empty(H).uniform_(math.log(dt_min), math.log(dt_max))


def _init_B(P: int, H: int, Vinv: np.ndarray) -> torch.Tensor:
    """Initialize B_tilde = Vinv @ B_raw.

    Args:
        P:    State size
        H:    Hidden dimension
        Vinv: (P, P) complex eigenvector inverse
    Returns:
        (P, H, 2) float tensor [real, imag]
    """
    std = 1.0 / math.sqrt(max(1, P))
    B_raw = np.random.normal(0, std, (P, H))
    VinvB = Vinv @ B_raw  # (P, H) complex
    return torch.stack(
        [torch.tensor(VinvB.real, dtype=torch.float32),
         torch.tensor(VinvB.imag, dtype=torch.float32)],
        dim=-1,
    )  # (P, H, 2)


def _init_C(H: int, P: int, V: np.ndarray) -> torch.Tensor:
    """Initialize C_tilde = C_raw @ V.

    Args:
        H: Hidden dimension
        P: State size
        V: (P, P) complex eigenvectors
    Returns:
        (H, P) complex tensor
    """
    std = 1.0 / math.sqrt(max(1, P))
    C_re = np.random.normal(0, std, (H, P))
    C_im = np.random.normal(0, std, (H, P))
    C = C_re + 1j * C_im
    CV = C @ V  # (H, P) complex
    return torch.complex(
        torch.tensor(CV.real, dtype=torch.float32),
        torch.tensor(CV.imag, dtype=torch.float32),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. Discretization
# ══════════════════════════════════════════════════════════════════════════════

def discretize_zoh(
    Lambda: torch.Tensor,
    B_tilde: torch.Tensor,
    Delta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Zero-order hold discretization.

    Args:
        Lambda:  (P,) complex diagonal state matrix
        B_tilde: (P, H) complex input matrix
        Delta:   (P,) float step sizes
    Returns:
        Lambda_bar (P,) complex, B_bar (P, H) complex
    """
    Identity = torch.ones(Lambda.shape[-1], device=Lambda.device, dtype=Lambda.dtype)
    Lambda_bar = torch.exp(Lambda * Delta)
    B_bar = (1.0 / Lambda * (Lambda_bar - Identity))[..., None] * B_tilde
    return Lambda_bar, B_bar


def discretize_bilinear(
    Lambda: torch.Tensor,
    B_tilde: torch.Tensor,
    Delta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bilinear (Tustin) discretization.

    Args:
        Lambda:  (P,) complex diagonal state matrix
        B_tilde: (P, H) complex input matrix
        Delta:   (P,) float step sizes
    Returns:
        Lambda_bar (P,) complex, B_bar (P, H) complex
    """
    Identity = torch.ones(Lambda.shape[-1], device=Lambda.device, dtype=Lambda.dtype)
    BL = 1.0 / (Identity - (Delta / 2.0) * Lambda)
    Lambda_bar = BL * (Identity + (Delta / 2.0) * Lambda)
    B_bar = (BL * Delta)[..., None] * B_tilde
    return Lambda_bar, B_bar


# ══════════════════════════════════════════════════════════════════════════════
# 4. SSM Apply
# ══════════════════════════════════════════════════════════════════════════════

def apply_ssm(
    Lambda_bar: torch.Tensor,
    B_bar: torch.Tensor,
    C_tilde: torch.Tensor,
    D: torch.Tensor,
    input_sequence: torch.Tensor,
    bidir: bool = False,
) -> torch.Tensor:
    """Apply discretized SSM to a single (non-batched) input sequence.

    Args:
        Lambda_bar:     (P,) or (L, P) complex
        B_bar:          (P, H) or (L, P, H) complex
        C_tilde:        (H, P) or (L, H, P) complex (2*P instead of P if bidir)
        D:              (H,) real direct feedthrough
        input_sequence: (L, H) real
        bidir:          concatenate forward and backward hidden states
    Returns:
        ys: (L, H) real
    """
    cinput = input_sequence.to(Lambda_bar.dtype)

    # Bu_elements: (L, P)
    if B_bar.ndim == 3:
        Bu_elements = torch.vmap(lambda B, u: B @ u)(B_bar, cinput)
    else:
        Bu_elements = torch.vmap(lambda u: B_bar @ u)(cinput)

    # Lambda_elements: (L, P)
    if Lambda_bar.ndim == 1:
        Lambda_elements = Lambda_bar.unsqueeze(0).expand(input_sequence.shape[0], -1)
    else:
        Lambda_elements = Lambda_bar

    _, xs = associative_scan(_binary_operator, (Lambda_elements, Bu_elements))

    if bidir:
        _, xs2 = associative_scan(_binary_operator, (Lambda_elements, Bu_elements), reverse=True)
        xs = torch.cat((xs, xs2), dim=-1)  # (L, 2*P)

    # Output: (L, H)
    if C_tilde.ndim == 3:
        # Time-varying C: (L, H, P) or (L, H, 2*P) if bidir
        Cx = torch.vmap(lambda C, x: (C @ x).real)(C_tilde, xs)
    else:
        # Static C: (H, P) or (H, 2*P) if bidir
        Cx = torch.vmap(lambda x: (C_tilde @ x).real)(xs)

    Du = torch.vmap(lambda u: D * u)(input_sequence)
    return Cx + Du


# ══════════════════════════════════════════════════════════════════════════════
# 5. GLU
# ══════════════════════════════════════════════════════════════════════════════

class GLU(nn.Module):
    """Gated Linear Unit: GLU(x) = w1(x) ⊙ sigmoid(w2(x))."""

    def __init__(self, d: int):
        super().__init__()
        self.w1 = nn.Linear(d, d)
        self.w2 = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w1(x) * torch.sigmoid(self.w2(x))


class ExpandedGLU(nn.Module):
    """GLU with feedforward expansion: H → H*ff_mult (gated) → H."""

    def __init__(self, d: int, ff_mult: float = 1.0):
        super().__init__()
        inner = int(d * ff_mult)
        self.w1 = nn.Linear(d, inner)
        self.w2 = nn.Linear(d, inner)
        self.proj = nn.Linear(inner, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.w1(x) * torch.sigmoid(self.w2(x)))


# ══════════════════════════════════════════════════════════════════════════════
# 6. GluEncoder   (matches JAX TIDES Encoder exactly)
# ══════════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class ComplexRMSNorm(nn.Module):
    """RMSNorm for complex-valued tensors. Normalizes by RMS of moduli."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(z.abs() ** 2, dim=-1, keepdim=True) + self.eps)
        return z / rms * self.weight


class GluEncoder(nn.Module):
    """Encoder: depth GLU residual layers (at d_in), then Linear(d_in → d_out).

    Matches JAX TIDES Encoder:
        for glu in hidden_layers: x = x + glu(x)
        return linear(x)
    GLU residuals operate at d_in dimension, then project to d_out.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        depth: int = 0,
        init_method: str = "default",
    ):
        super().__init__()
        self.glu_layers = nn.ModuleList([GLU(d_in) for _ in range(depth)])
        self.linear = nn.Linear(d_in, d_out)
        if init_method == "zeros":
            nn.init.zeros_(self.linear.weight)
            # bias left at default (will be set externally for lambda_proj/bc_proj)
        elif init_method == "random":
            nn.init.normal_(self.linear.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for glu in self.glu_layers:
            x = x + glu(x)   # residual at d_in
        return self.linear(x)


# ══════════════════════════════════════════════════════════════════════════════
# 6b. LowRankHead   (factored projection for B/C)
# ══════════════════════════════════════════════════════════════════════════════

class LowRankHead(nn.Module):
    """Factored projection head: Linear(d_in, rank) -> Linear(rank, d_out)."""

    def __init__(self, d_in: int, d_out: int, rank: int, init_method: str = "zeros"):
        super().__init__()
        self.down = nn.Linear(d_in, rank)
        self.up = nn.Linear(rank, d_out)
        if init_method == "zeros":
            nn.init.zeros_(self.up.weight)
        elif init_method == "random":
            nn.init.normal_(self.up.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


# ══════════════════════════════════════════════════════════════════════════════
# 7. TIDESSSM
# ══════════════════════════════════════════════════════════════════════════════

class TIDESSSM(nn.Module):
    """Input-Dependent S5 SSM layer (single sequence, no batch dim).

    Faithfully follows the JAX TIDES layer with two additions:
    - Per-timestep step_scale for irregular sampling
    - PyTorch autograd-compatible implementation

    The state size P equals ssm_size (no conjugate-symmetry halving).
    """

    def __init__(
        self,
        ssm_size: int,
        blocks: int,
        H: int,
        clip_eigs: bool = False,
        discretization: Literal["zoh", "bilinear"] = "zoh",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        lambda_re_mode: Literal["lti", "input_dependent"] = "input_dependent",
        lambda_im_mode: Literal["lti", "input_dependent"] = "lti",
        bc_mode: Literal["lti", "input_dependent"] = "input_dependent",
        learn_lambda: Literal["standard", "exp", "stable", "softplus"] = "standard",
        bidir: bool = False,
        lambda_encoder_depth: int = 0,
        bc_rank: int = 8,
        proj_init_method: str = "zeros",
        proj_norm: Optional[Literal["rmsnorm"]] = "rmsnorm",
        step_mode: Literal["lti", "input_dependent"] = "lti",
    ):
        super().__init__()
        for name, m in (('lambda_re_mode', lambda_re_mode),
                        ('lambda_im_mode', lambda_im_mode),
                        ('bc_mode', bc_mode),
                        ('step_mode', step_mode)):
            if m not in ('lti', 'input_dependent'):
                raise ValueError(
                    f"{name} must be 'lti' or 'input_dependent', got {m!r}")
        if bc_mode == "input_dependent" and bc_rank < 1:
            raise ValueError(f"bc_rank must be >= 1, got {bc_rank}")
        if proj_norm not in ("rmsnorm", None):
            raise ValueError(f"proj_norm must be 'rmsnorm' or None, got {proj_norm!r}")
        self.H = H
        self.clip_eigs = clip_eigs
        self.discretization = discretization
        self.lambda_re_mode = lambda_re_mode
        self.lambda_im_mode = lambda_im_mode
        self.bc_mode = bc_mode
        self.learn_lambda = learn_lambda
        self.bidir = bidir
        self.step_mode = step_mode

        # ── HiPPO initialization ──────────────────────────────────────────────
        block_size = ssm_size // blocks
        Lambda_np, _, _, V_np = _make_DPLR_HiPPO(block_size)
        P = ssm_size
        self.P = P
        Vc_np = V_np.conj().T

        # Block-diagonal construction
        V_blocks = scipy.linalg.block_diag(*([V_np] * blocks))      # (P, P)
        Vinv_blocks = scipy.linalg.block_diag(*([Vc_np] * blocks))  # (P, P)
        Lambda_np = np.tile(Lambda_np, blocks)  # (P,)

        # ── Lambda parameters ─────────────────────────────────────────────────
        if learn_lambda == "exp":
            Lambda_re_param = np.log(-Lambda_np.real)
        elif learn_lambda == "stable":
            w2 = np.maximum(0.0, -1.0 / Lambda_np.real - 0.5)
            Lambda_re_param = np.sqrt(w2)
        elif learn_lambda == "softplus":
            Lambda_re_param = np.log(np.exp(-Lambda_np.real) - 1.0)
        else:
            Lambda_re_param = Lambda_np.real

        self.Lambda_re = nn.Parameter(torch.tensor(Lambda_re_param, dtype=torch.float32))
        self.Lambda_im = nn.Parameter(torch.tensor(Lambda_np.imag, dtype=torch.float32))

        # ── B and C parameters ────────────────────────────────────────────────
        B_init = _init_B(P, H, Vinv_blocks)   # (P, H, 2) float
        self.B = nn.Parameter(B_init)

        # C matrix: (H, P) for unidirectional, (H, 2*P) for bidirectional
        C_init = _init_C(H, P, V_blocks)      # (H, P) complex
        if bidir:
            # Initialize separate C for backward pass and concatenate
            C_init_bwd = _init_C(H, P, V_blocks)
            C_init = torch.cat([C_init, C_init_bwd], dim=-1)  # (H, 2*P)
        self.C = nn.Parameter(torch.view_as_real(C_init.contiguous()))  # (H, P, 2) or (H, 2*P, 2)

        # ── D (direct feedthrough) ────────────────────────────────────────────
        self.D = nn.Parameter(torch.randn(H))

        # ── Learnable timescale ───────────────────────────────────────────────
        self.log_step = nn.Parameter(_init_log_steps(P, dt_min, dt_max))

        # ── Input-dependent step (Mamba-style surrogate, "Mamba_S") ───────────
        # step = softplus(W [x, Δ] + b): Δ becomes one more input feature rather
        # than the physical interval that scales the discretization.  Starts at
        # step = 1 for every state (softplus⁻¹(1) = 0.5413).
        if step_mode == "input_dependent":
            self.step_proj = nn.Linear(H + 1, P)
            with torch.no_grad():
                nn.init.zeros_(self.step_proj.weight)
                self.step_proj.bias.fill_(0.5413)
        else:
            self.step_proj = None

        # ── Input-dependent Lambda REAL projector ─────────────────────────────
        # Bias = the static parameter, so the projector starts at HiPPO.
        if lambda_re_mode == "input_dependent":
            self.lambda_re_proj = GluEncoder(
                H, P, depth=lambda_encoder_depth, init_method=proj_init_method
            )
            with torch.no_grad():
                self.lambda_re_proj.linear.bias.copy_(
                    torch.tensor(Lambda_re_param, dtype=torch.float32))
        else:
            self.lambda_re_proj = None

        # ── Input-dependent Lambda IMAG projector ─────────────────────────────
        if lambda_im_mode == "input_dependent":
            self.lambda_im_proj = GluEncoder(
                H, P, depth=lambda_encoder_depth, init_method=proj_init_method
            )
            with torch.no_grad():
                self.lambda_im_proj.linear.bias.copy_(
                    torch.tensor(Lambda_np.imag, dtype=torch.float32))
        else:
            self.lambda_im_proj = None

        # ── Input-dependent BC projectors (low-rank heads) ────────────────────
        # Bias = flattened HiPPO B/C values, so the SSM starts at HiPPO.
        C_dim = 2 * P if bidir else P
        if bc_mode == "input_dependent":
            self.b_proj = LowRankHead(H, P * H * 2, bc_rank, init_method=proj_init_method)
            self.c_proj = LowRankHead(H, H * C_dim * 2, bc_rank, init_method=proj_init_method)
            with torch.no_grad():
                self.b_proj.up.bias.copy_(B_init.reshape(-1))
                self.c_proj.up.bias.copy_(self.C.data.reshape(-1))
        else:
            self.b_proj = None
            self.c_proj = None

        # ── Optional RMSNorm on projected parameters ─────────────────────────
        self.lambda_re_norm = (
            RMSNorm(P) if proj_norm == "rmsnorm" and lambda_re_mode == "input_dependent"
            else None)
        self.lambda_im_norm = (
            RMSNorm(P) if proj_norm == "rmsnorm" and lambda_im_mode == "input_dependent"
            else None)
        self.b_norm = (
            ComplexRMSNorm(P) if proj_norm == "rmsnorm" and bc_mode == "input_dependent"
            else None)
        self.c_norm = (
            ComplexRMSNorm(C_dim) if proj_norm == "rmsnorm" and bc_mode == "input_dependent"
            else None)

    # ── Lambda helpers ────────────────────────────────────────────────────────

    def _apply_re_transform(self, raw_re: torch.Tensor) -> torch.Tensor:
        """Apply the learn_lambda reparameterization and optional clipping to Re(Lambda)."""
        if self.learn_lambda == "exp":
            re = -torch.exp(raw_re)
        elif self.learn_lambda == "stable":
            re = -1.0 / (raw_re ** 2 + 0.5)
        elif self.learn_lambda == "softplus":
            re = -F.softplus(raw_re)
        else:
            re = raw_re
        if self.clip_eigs:
            re = torch.clamp(re, max=-1e-4)
        return re

    def _get_lambda(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Lambda with independent control over real and imaginary parts.

        Args:
            x: (L, H) input sequence
        Returns:
            Lambda: (P,) complex if both lti, (L, P) complex otherwise
        """
        # --- Real part ---
        if self.lambda_re_mode == "lti":
            Lambda_re = self._apply_re_transform(self.Lambda_re)  # (P,)
        else:  # input_dependent
            f_re = torch.vmap(self.lambda_re_proj)(x)  # (L, P)
            if self.lambda_re_norm is not None:
                f_re = torch.vmap(self.lambda_re_norm)(f_re)
            Lambda_re = self._apply_re_transform(f_re)

        # --- Imaginary part ---
        if self.lambda_im_mode == "lti":
            Lambda_im = self.Lambda_im  # (P,)
        else:  # input_dependent
            Lambda_im = torch.vmap(self.lambda_im_proj)(x)  # (L, P)
            if self.lambda_im_norm is not None:
                Lambda_im = torch.vmap(self.lambda_im_norm)(Lambda_im)

        return torch.complex(Lambda_re, Lambda_im)

    # ── B/C helpers ───────────────────────────────────────────────────────────

    def _get_BC(self, x: torch.Tensor):
        """Compute B and C (static or input-dependent).

        Args:
            x: (L, H) input sequence
        Returns:
            B_tilde: (P, H) complex if lti, (L, P, H) otherwise
            C_tilde: (H, C_dim) complex if lti, (L, H, C_dim) otherwise
                     (C_dim = 2*P if bidir else P)
        """
        if self.bc_mode == "lti":
            return (torch.view_as_complex(self.B.contiguous()),   # (P, H)
                    torch.view_as_complex(self.C.contiguous()))   # (H, C_dim)

        L = x.shape[0]
        C_dim = 2 * self.P if self.bidir else self.P

        # Separate low-rank projections for B and C
        B_flat = torch.vmap(self.b_proj)(x)
        C_flat = torch.vmap(self.c_proj)(x)
        B_dyn = torch.view_as_complex(
            B_flat.reshape(L, self.P, self.H, 2).contiguous()
        )  # (L, P, H)
        C_dyn = torch.view_as_complex(
            C_flat.reshape(L, self.H, C_dim, 2).contiguous()
        )  # (L, H, C_dim)
        if self.b_norm is not None:
            B_dyn = torch.vmap(lambda b: torch.vmap(self.b_norm)(b.T).T)(B_dyn)
        if self.c_norm is not None:
            C_dyn = torch.vmap(lambda c: torch.vmap(self.c_norm)(c))(C_dyn)

        return B_dyn, C_dyn

    # ── Core prepare + forward ────────────────────────────────────────────────

    def _prepare(self, x: torch.Tensor, step_scale=1.0):
        """Compute discretized (Lambda_bar, B_bar, C_tilde).

        Args:
            x:          (L, H) input sequence
            step_scale: float or (L,) tensor of per-timestep scales
        Returns:
            Lambda_bar: (P,) or (L, P) complex
            B_bar:      (P, H) or (L, P, H) complex
            C_tilde:    (H, P) or (L, H, P) complex
        """
        Lambda = self._get_lambda(x)       # (P,) or (L, P)
        B_tilde, C_tilde = self._get_BC(x)  # static or (L, ...)

        # Compute step: (P,) or (L, P)
        if self.step_mode == "input_dependent":
            if torch.is_tensor(step_scale) and step_scale.ndim > 0:
                dt_feat = step_scale.unsqueeze(-1)  # (L, 1)
            else:
                dt_feat = torch.full((x.shape[0], 1), float(step_scale),
                                     device=x.device, dtype=x.dtype)
            step = F.softplus(self.step_proj(torch.cat([x, dt_feat], dim=-1)))  # (L, P)
        else:
            step_base = torch.exp(self.log_step)  # (P,)
            if not torch.is_tensor(step_scale) or step_scale.ndim == 0:
                step = float(step_scale) * step_base   # (P,)
            else:
                step = step_scale[:, None] * step_base[None, :]  # (L, P)

        # Detect time-varying dimensions from tensor rank
        disc_fn = discretize_zoh if self.discretization == "zoh" else discretize_bilinear
        lambda_tv = Lambda.ndim == 2
        bc_tv = B_tilde.ndim == 3
        step_tv = torch.is_tensor(step) and step.ndim == 2

        if not lambda_tv and not bc_tv and not step_tv:
            # All static: single discretization call
            Lambda_bar, B_bar = disc_fn(Lambda, B_tilde, step)
        else:
            # At least one is time-varying: vmap over L
            L = x.shape[0]
            Lambda_L = Lambda if lambda_tv else Lambda.unsqueeze(0).expand(L, -1)
            B_L = B_tilde if bc_tv else B_tilde.unsqueeze(0).expand(L, -1, -1)
            step_L = step if step_tv else step.unsqueeze(0).expand(L, -1)
            Lambda_bar, B_bar = torch.vmap(disc_fn)(Lambda_L, B_L, step_L)

        return Lambda_bar, B_bar, C_tilde

    def forward(self, x: torch.Tensor, step_scale=1.0) -> torch.Tensor:
        """Forward pass for a single (non-batched) sequence.

        Args:
            x:          (L, H) input sequence
            step_scale: float or (L,) tensor of per-timestep step sizes
        Returns:
            y: (L, H)
        """
        Lambda_bar, B_bar, C_tilde = self._prepare(x, step_scale)
        return apply_ssm(Lambda_bar, B_bar, C_tilde, self.D, x, self.bidir)


# ══════════════════════════════════════════════════════════════════════════════
# 8. TIDESBlock
# ══════════════════════════════════════════════════════════════════════════════

class TIDESBlock(nn.Module):
    """TIDES Block following JAX structure exactly.

    JAX order: BatchNorm → TIDESLayer → GELU → Dropout → GLU → Dropout → residual

    Uses BatchNorm1d with affine=False (matches JAX channelwise_affine=False).
    SSM is applied per-sample via vmap; GLU and dropout operate on full batch.
    """

    def __init__(
        self,
        ssm_size: int,
        blocks: int,
        H: int,
        clip_eigs: bool = False,
        discretization: Literal["zoh", "bilinear"] = "zoh",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        lambda_re_mode: Literal["lti", "input_dependent"] = "input_dependent",
        lambda_im_mode: Literal["lti", "input_dependent"] = "lti",
        bc_mode: Literal["lti", "input_dependent"] = "input_dependent",
        learn_lambda: Literal["standard", "exp", "stable", "softplus"] = "standard",
        bidir: bool = False,
        drop_rate: float = 0.0,
        lambda_encoder_depth: int = 0,
        bc_rank: int = 8,
        ff_mult: float = 1.0,
        proj_init_method: str = "zeros",
        proj_norm: Optional[Literal["rmsnorm"]] = "rmsnorm",
        step_mode: Literal["lti", "input_dependent"] = "lti",
    ):
        super().__init__()
        # BatchNorm without learnable affine: matches JAX channelwise_affine=False
        self.norm = nn.BatchNorm1d(H, affine=False)
        self.ssm = TIDESSSM(
            ssm_size=ssm_size,
            blocks=blocks,
            H=H,
            clip_eigs=clip_eigs,
            discretization=discretization,
            dt_min=dt_min,
            dt_max=dt_max,
            lambda_re_mode=lambda_re_mode,
            lambda_im_mode=lambda_im_mode,
            bc_mode=bc_mode,
            learn_lambda=learn_lambda,
            bidir=bidir,
            lambda_encoder_depth=lambda_encoder_depth,
            bc_rank=bc_rank,
            proj_init_method=proj_init_method,
            proj_norm=proj_norm,
            step_mode=step_mode,
        )
        self.glu = ExpandedGLU(H, ff_mult)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor, step_scale=1.0) -> torch.Tensor:
        """
        Args:
            x:          (B, L, H)
            step_scale: float or (B, L) tensor of per-timestep step sizes
        Returns:
            x: (B, L, H)
        """
        skip = x

        # BatchNorm1d expects (B, H, L); normalize over B and L per channel H
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)  # (B, L, H)

        # Apply SSM per sample in batch via vmap
        if not torch.is_tensor(step_scale) or step_scale.ndim < 2:
            # Scalar or per-batch step: broadcast to all timesteps
            x = torch.vmap(lambda s: self.ssm(s, step_scale))(x)
        else:
            # Per-timestep step_scale: (B, L) → each sample gets (L,)
            x = torch.vmap(lambda s, ss: self.ssm(s, ss))(x, step_scale)

        x = self.dropout(F.gelu(x))
        x = self.glu(x)         # GLU operates on (B, L, H) via Linear on last dim
        x = self.dropout(x)
        return skip + x


# ══════════════════════════════════════════════════════════════════════════════
# 9. TIDES (top-level model)
# ══════════════════════════════════════════════════════════════════════════════

class TIDES(nn.Module):
    """TIDES encoder: (B, L, d_input) -> (B, L, d_hidden).

    Architecture follows JAX TIDES:
        GluEncoder(d_input → d_hidden)
        → [TIDESBlock × num_blocks]
        → (B, L, d_hidden) features

    The defaults are the TIDES model of the paper: input-dependent Re(Λ),
    B and C, static Im(Λ).  The output head is not included here:
    TIDESClassifier and TIDESForecastingModel add one.

    Args:
        d_input:              Input feature dimension
        d_hidden:             Hidden dimension H
        ssm_size:             Total SSM state size P
        ssm_blocks:           Diagonal blocks for HiPPO initialization
        num_blocks:           Number of TIDES blocks
        clip_eigs:            Clip eigenvalues to be strictly negative
        discretization:       'zoh' or 'bilinear'
        dt_min / dt_max:      Log-step initialization range
        lambda_re_mode:       'lti' or 'input_dependent' (decay)
        lambda_im_mode:       'lti' or 'input_dependent' (oscillation)
        bc_mode:              'lti' or 'input_dependent'
        learn_lambda:         'standard', 'exp', 'stable' or 'softplus'
        bidir:                Weight-tied bidirectional scan
        drop_rate:            Dropout rate inside blocks
        encoder_depth:        GLU residual layers in the input encoder
        lambda_encoder_depth: GLU residual layers in the Lambda projector
        bc_rank:              Rank of the low-rank B/C projector heads (>= 1)
        ff_mult:              GLU expansion factor in each block
        proj_init_method:     'zeros' or 'random' init of the projector weights
        proj_norm:            'rmsnorm' on the projector outputs, or None for none
        step_mode:            'lti' (TIDES: step = Δ · exp(log_step)) or
                              'input_dependent' (Mamba-style surrogate:
                              step = softplus(W [x, Δ] + b))
    """

    def __init__(
        self,
        d_input: int,
        d_hidden: int,
        ssm_size: int,
        ssm_blocks: int,
        num_blocks: int,
        clip_eigs: bool = False,
        discretization: Literal["zoh", "bilinear"] = "zoh",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        lambda_re_mode: Literal["lti", "input_dependent"] = "input_dependent",
        lambda_im_mode: Literal["lti", "input_dependent"] = "lti",
        bc_mode: Literal["lti", "input_dependent"] = "input_dependent",
        learn_lambda: Literal["standard", "exp", "stable", "softplus"] = "standard",
        bidir: bool = False,
        drop_rate: float = 0.0,
        encoder_depth: int = 0,
        lambda_encoder_depth: int = 0,
        bc_rank: int = 8,
        ff_mult: float = 1.0,
        proj_init_method: str = "zeros",
        proj_norm: Optional[Literal["rmsnorm"]] = "rmsnorm",
        step_mode: Literal["lti", "input_dependent"] = "lti",
    ):
        super().__init__()
        # Input encoder: GLU residuals at d_input, then project to d_hidden
        self.encoder = GluEncoder(d_input, d_hidden, depth=encoder_depth)
        self.blocks = nn.ModuleList([
            TIDESBlock(
                ssm_size=ssm_size,
                blocks=ssm_blocks,
                H=d_hidden,
                clip_eigs=clip_eigs,
                discretization=discretization,
                dt_min=dt_min,
                dt_max=dt_max,
                lambda_re_mode=lambda_re_mode,
                lambda_im_mode=lambda_im_mode,
                bc_mode=bc_mode,
                learn_lambda=learn_lambda,
                bidir=bidir,
                drop_rate=drop_rate,
                lambda_encoder_depth=lambda_encoder_depth,
                bc_rank=bc_rank,
                ff_mult=ff_mult,
                proj_init_method=proj_init_method,
                proj_norm=proj_norm,
                step_mode=step_mode,
            )
            for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor, step_scale=1.0) -> torch.Tensor:
        """
        Args:
            x:          (B, L, d_input) input sequence
            step_scale: time since the previous observation: a float
                        (1.0 = regular sampling), an (L,) or a (B, L) tensor
        Returns:
            (B, L, d_hidden) feature sequence
        """
        x = self.encoder(x)           # (B, L, d_hidden)
        for block in self.blocks:
            x = block(x, step_scale=step_scale)
        return x                       # (B, L, d_hidden)


# ══════════════════════════════════════════════════════════════════════════════
# 10. TIDESClassifier (UEA classification head)
# ══════════════════════════════════════════════════════════════════════════════

class TIDESClassifier(nn.Module, PyTorchModelHubMixin, **hub_kwargs("time-series-classification")):
    """TIDES backbone + mean-pool + linear head for sequence classification.

    Accepts the same step_scale as TIDES: float, (L,), or (B, L) tensor,
    the time since the previous observation.  For a sequence subsampled
    from a regular grid, step_scale_from_indices(kept_indices) gives it.

    Supports save_pretrained / from_pretrained / push_to_hub (Hugging Face Hub).
    """

    def __init__(
        self,
        d_input: int,
        num_classes: int,
        d_hidden: int = 64,
        ssm_size: int = 64,
        ssm_blocks: int = 2,
        num_blocks: int = 2,
        clip_eigs: bool = False,
        discretization: Literal["zoh", "bilinear"] = "zoh",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        lambda_re_mode: Literal["lti", "input_dependent"] = "input_dependent",
        lambda_im_mode: Literal["lti", "input_dependent"] = "lti",
        bc_mode: Literal["lti", "input_dependent"] = "input_dependent",
        learn_lambda: Literal["standard", "exp", "stable", "softplus"] = "standard",
        bidir: bool = False,
        drop_rate: float = 0.0,
        encoder_depth: int = 0,
        lambda_encoder_depth: int = 0,
        bc_rank: int = 8,
        ff_mult: float = 1.0,
        proj_init_method: str = "zeros",
        proj_norm: Optional[Literal["rmsnorm"]] = "rmsnorm",
        step_mode: Literal["lti", "input_dependent"] = "lti",
    ):
        super().__init__()
        self.backbone = TIDES(
            d_input=d_input, d_hidden=d_hidden, ssm_size=ssm_size,
            ssm_blocks=ssm_blocks, num_blocks=num_blocks,
            clip_eigs=clip_eigs,
            discretization=discretization, dt_min=dt_min, dt_max=dt_max,
            lambda_re_mode=lambda_re_mode, lambda_im_mode=lambda_im_mode,
            bc_mode=bc_mode, learn_lambda=learn_lambda, bidir=bidir,
            drop_rate=drop_rate, encoder_depth=encoder_depth,
            lambda_encoder_depth=lambda_encoder_depth, bc_rank=bc_rank,
            ff_mult=ff_mult,
            proj_init_method=proj_init_method, proj_norm=proj_norm,
            step_mode=step_mode,
        )
        self.head = nn.Linear(d_hidden, num_classes)

    def forward(self, x: torch.Tensor, step_scale=1.0) -> torch.Tensor:
        """Args: x (B, L, d_input); step_scale float or (L,) or (B, L).

        Returns: logits (B, num_classes).
        """
        h = self.backbone(x, step_scale=step_scale)  # (B, L, d_hidden)
        return self.head(h.mean(dim=1))               # (B, num_classes)


# ══════════════════════════════════════════════════════════════════════════════
# 11. Step-scale utility
# ══════════════════════════════════════════════════════════════════════════════

def step_scale_from_indices(keep_indices, device=None) -> torch.Tensor:
    """Compute per-timestep step sizes from randomly kept time indices.

    For a sequence subsampled at keep_indices from a uniform grid, the step
    at position i is the gap between consecutive kept indices:
        step[0] = 1
        step[i] = keep_indices[i] - keep_indices[i - 1]

    When keep_indices = [0, 1, 2, ..., L-1] (no drop), all steps = 1, identical
    to the default uniform step_scale = 1.0.
    """
    keep = list(keep_indices)
    gaps = torch.ones(len(keep), dtype=torch.float32)
    for i in range(1, len(keep)):
        gaps[i] = keep[i] - keep[i - 1]
    return gaps if device is None else gaps.to(device)
