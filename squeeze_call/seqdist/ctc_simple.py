# AUTOGENERATED! DO NOT EDIT! File to edit: notebooks/04_CTC_loss_simple.ipynb (unless otherwise specified).

__all__ = [
    "device",
    "generate_sample_inputs",
    "logZ_fwd",
    "dot",
    "LogZ",
    "logZ_py",
    "mean",
    "logZ_cupy",
    "viterbi_alignments",
    "soft_alignments",
    "cupy_funcs",
    "cupy_funcs_loop",
]

# Cell
import numpy as np
import cupy as cp
import torch
import torch.nn as nn
from collections import namedtuple
from .utils import *

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Cell
def generate_sample_inputs(T, N, L_min, L_max, device=device):
    """
    Args:
        T: number of time steps
        N: batch size
        L_min, L_max: bounds on target length
    """
    stay_scores = torch.rand(T, N, L_max, device=device, requires_grad=True)
    move_scores = torch.rand(T, N, L_max - 1, device=device, requires_grad=True)
    target_lengths = torch.randint(L_min, L_max + 1, (N,), device=device)
    return stay_scores, move_scores, target_lengths


# Cell
from torch.nn.functional import pad
from .core import semiring, Log, Max


def logZ_fwd(stay_scores, move_scores, target_lengths, S=Log):
    T, N, L = stay_scores.shape
    alpha_0 = stay_scores.new_full((N, L), S.zero)
    alpha_0[:, 0] = S.one
    beta_T = stay_scores.new_full((N, L), S.zero)
    beta_T[torch.arange(N), target_lengths - 1] = S.one
    move_scores = pad(move_scores, (1, 0), value=S.zero)
    a = pad(alpha_0, (1, 0), value=S.zero)
    for t in range(0, stay_scores.size(0)):
        a[:, 1:] = S.sum(
            torch.stack(
                [S.mul(stay_scores[t], a[:, 1:]), S.mul(move_scores[t], a[:, :-1])]
            ),
            dim=0,
        )
    return S.sum(S.mul(a[:, 1:], beta_T), dim=1)


# Cell
def _simple_lattice_fwd_bwd(
    alpha, beta_T, beta_stay, beta_move, stay_scores, move_scores, S=Log
):
    T = alpha.size(0) - 1
    move_scores = pad(move_scores, (1, 1), value=S.zero)
    a = pad(alpha[0], (1, 0), value=S.zero)
    for t in range(0, T):
        a[:, 1:] = S.sum(
            torch.stack(
                [
                    S.mul(stay_scores[t], a[:, 1:]),
                    S.mul(move_scores[t, :, :-1], a[:, :-1]),
                ]
            ),
            dim=0,
        )
        alpha[t + 1] = a[:, 1:]

    b = pad(beta_T, (0, 1), value=S.zero)
    for t in range(T, 0, -1):
        beta_stay[t - 1] = S.mul(b[:, :-1], stay_scores[t - 1])
        beta_move[t - 1] = S.mul(b[:, 1:], move_scores[t - 1, :, 1:])
        b[:, :-1] = S.sum(torch.stack([beta_stay[t - 1], beta_move[t - 1]]), dim=0)


def dot(x, y, S=Log, dim=-1):
    return S.sum(S.mul(x, y), dim=dim)


class LogZ(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, stay_scores, move_scores, target_lengths, fwd_bwd_impl, S: semiring
    ):
        T, N, L = stay_scores.shape

        alpha = stay_scores.new_full((T + 1, N, L), S.zero)
        alpha[0, :, 0] = S.one

        beta_stay = stay_scores.new_full((T, N, L), S.zero)
        beta_move = stay_scores.new_full((T, N, L), S.zero)
        beta_T = stay_scores.new_full((N, L), S.zero)
        beta_T[torch.arange(N), target_lengths - 1] = S.one

        fwd_bwd_impl(alpha, beta_T, beta_stay, beta_move, stay_scores, move_scores, S)

        g = S.dsum(
            torch.cat(
                [S.mul(alpha[:-1], beta_stay), S.mul(alpha[:-1], beta_move)], dim=2
            ),
            dim=2,
        )

        ctx.save_for_backward(g.reshape(T, N, 2, L))
        return dot(alpha[-1], beta_T, S)

    @staticmethod
    def backward(ctx, grad):
        g = ctx.saved_tensors[0] * grad[None, :, None, None]
        return g[:, :, 0], g[:, :, 1, :-1], None, None, None


def logZ_py(stay_scores, move_scores, target_lengths):
    return LogZ.apply(
        stay_scores, move_scores, target_lengths, _simple_lattice_fwd_bwd, Log
    )


# Cell
mean = lambda f: (lambda *xs: f(*xs).mean())

# Cell
from .utils import *
import cupy as cp

cupy_funcs = {
    (torch.float32, Log): load_cupy_func(
        "cuda/ctc_simple.cu",
        "fwd_bwd_logspace",
        FLOAT="float",
        SUM="logsumexp2",
        MUL="add",
        ZERO="{:E}".format(Log.zero),
    ),
    (torch.float64, Log): load_cupy_func(
        "cuda/ctc_simple.cu",
        "fwd_bwd_logspace",
        FLOAT="double",
        SUM="logsumexp2",
        MUL="add",
        ZERO="{:E}".format(Log.zero),
    ),
    (torch.float32, Max): load_cupy_func(
        "cuda/ctc_simple.cu",
        "fwd_bwd_logspace",
        FLOAT="float",
        SUM="max2",
        MUL="add",
        ZERO="{:E}".format(Max.zero),
    ),
    (torch.float64, Max): load_cupy_func(
        "cuda/ctc_simple.cu",
        "fwd_bwd_logspace",
        FLOAT="double",
        SUM="max2",
        MUL="add",
        ZERO="{:E}".format(Max.zero),
    ),
}

cupy_funcs_loop = {
    (torch.float32, Log): load_cupy_func(
        "cuda/ctc_simple.cu",
        "fwd_bwd_logspace_loop",
        FLOAT="float",
        SUM="logsumexp2",
        MUL="add",
        ZERO="{:E}".format(Log.zero),
    ),
    (torch.float64, Log): load_cupy_func(
        "cuda/ctc_simple.cu",
        "fwd_bwd_logspace_loop",
        FLOAT="double",
        SUM="logsumexp2",
        MUL="add",
        ZERO="{:E}".format(Log.zero),
    ),
    (torch.float32, Max): load_cupy_func(
        "cuda/ctc_simple.cu",
        "fwd_bwd_logspace_loop",
        FLOAT="float",
        SUM="max2",
        MUL="add",
        ZERO="{:E}".format(Max.zero),
    ),
    (torch.float64, Max): load_cupy_func(
        "cuda/ctc_simple.cu",
        "fwd_bwd_logspace_loop",
        FLOAT="double",
        SUM="max2",
        MUL="add",
        ZERO="{:E}".format(Max.zero),
    ),
}


def _simple_lattice_fwd_bwd_cupy(
    alpha, beta_T, beta_stay, beta_move, stay_scores, move_scores, S: semiring
):
    T, N, L = stay_scores.shape
    if L > 1024:  # exceeds max threads per block
        return _simple_lattice_fwd_bwd_cupy_loop(
            alpha, beta_T, beta_stay, beta_move, stay_scores, move_scores, S
        )
    _bytes = 8 if (stay_scores.dtype == torch.float64) else 4
    with cp.cuda.Device(stay_scores.device.index):
        cupy_funcs[(stay_scores.dtype, S)](
            grid=(N, 2, 1),
            block=(L, 1, 1),
            shared_mem=2 * _bytes * L,
            args=(
                alpha.data_ptr(),
                beta_T.data_ptr(),
                beta_stay.data_ptr(),
                beta_move.data_ptr(),
                stay_scores.data_ptr(),
                move_scores.data_ptr(),
                T,
                N,
                L,
            ),
        )


def _simple_lattice_fwd_bwd_cupy_loop(
    alpha,
    beta_T,
    beta_stay,
    beta_move,
    stay_scores,
    move_scores,
    S: semiring,
    max_block_size=1024,
):
    T, N, L = stay_scores.shape
    block_size = min(L, max_block_size)
    beta = alpha.new_full(alpha.shape, S.zero)
    beta[-1] = beta_T
    with cp.cuda.Device(stay_scores.device.index):
        cupy_funcs_loop[(stay_scores.dtype, S)](
            grid=(N, 2, 1),
            block=(block_size, 1, 1),
            args=(
                alpha.data_ptr(),
                beta.data_ptr(),
                beta_stay.data_ptr(),
                beta_move.data_ptr(),
                stay_scores.data_ptr(),
                move_scores.data_ptr(),
                T,
                N,
                L,
            ),
        )


def logZ_cupy(stay_scores, move_scores, target_lengths, S: semiring = Log):
    return LogZ.apply(
        stay_scores, move_scores, target_lengths, _simple_lattice_fwd_bwd_cupy, S
    )


def viterbi_alignments(stay_scores, move_scores, target_lengths):
    target_lengths = target_lengths.to(stay_scores.device)
    stay_scores, move_scores = (
        stay_scores.detach().requires_grad_(),
        move_scores.detach().requires_grad_(),
    )
    logZ_cupy(stay_scores, move_scores, target_lengths, Max).sum().backward()
    alignments = stay_scores.grad.clone()
    alignments[:, :, :-1] += move_scores.grad
    return alignments


def soft_alignments(stay_scores, move_scores, target_lengths, beta=1.0):
    target_lengths = target_lengths.to(stay_scores.device)
    if beta != 1.0:
        stay_scores, move_scores = stay_scores * beta, move_scores * beta
    stay_scores, move_scores = (
        stay_scores.detach().requires_grad_(),
        move_scores.detach().requires_grad_(),
    )
    logZ_cupy(stay_scores, move_scores, target_lengths, Log).sum().backward()
    alignments = stay_scores.grad.clone()
    alignments[:, :, :-1] += move_scores.grad
    return alignments