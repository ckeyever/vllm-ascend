# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fla/ops/l2norm.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


MAX_BLOCK_ELEMENTS = 8192
DEFAULT_MBLOCK = 128
SPLIT_N_BLOCK = 4096


@triton.jit(do_not_specialize=["eps", "M", "NUM_CHUNKS"])
def l2norm_fwd_kernel2_loop(X, Y, eps, M, N: tl.constexpr, MBLOCK: tl.constexpr, NUM_CHUNKS):
    base_row = tl.program_id(0) * (NUM_CHUNKS * MBLOCK)
    rindex = tl.arange(0, N)[None, :]

    for chunk in range(NUM_CHUNKS):
        row_idx = base_row + chunk * MBLOCK + tl.arange(0, MBLOCK)[:, None]
        xmask = row_idx < M

        xs = tl.load(X + (rindex + N * row_idx), mask=xmask, other=0.0).to(tl.float32)
        square = xs * xs
        square_sum = tl.sum(square, 1)[:, None]
        rsqrt = tl.rsqrt(square_sum + eps)

        tl.store(Y + (rindex + N * row_idx), xs * rsqrt, xmask)


@triton.jit(do_not_specialize=["eps", "M", "NUM_CHUNKS"])
def l2norm_fwd_kernel2_loop_split_n(
    X,
    Y,
    eps,
    M,
    N: tl.constexpr,
    MBLOCK: tl.constexpr,
    NBLOCK: tl.constexpr,
    NUM_N_CHUNKS: tl.constexpr,
    NUM_CHUNKS,
):
    base_row = tl.program_id(0) * (NUM_CHUNKS * MBLOCK)
    n_offsets = tl.arange(0, NBLOCK)[None, :]

    for chunk in range(NUM_CHUNKS):
        row_idx = base_row + chunk * MBLOCK + tl.arange(0, MBLOCK)[:, None]
        xmask = row_idx < M
        square_sum = tl.zeros((MBLOCK, 1), dtype=tl.float32)

        for n_chunk in range(NUM_N_CHUNKS):
            rindex = n_chunk * NBLOCK + n_offsets
            nmask = rindex < N
            mask = xmask & nmask
            xs = tl.load(X + (rindex + N * row_idx), mask=mask, other=0.0).to(tl.float32)
            square_sum += tl.sum(xs * xs, 1)[:, None]

        rsqrt = tl.rsqrt(square_sum + eps)

        for n_chunk in range(NUM_N_CHUNKS):
            rindex = n_chunk * NBLOCK + n_offsets
            nmask = rindex < N
            mask = xmask & nmask
            xs = tl.load(X + (rindex + N * row_idx), mask=mask, other=0.0).to(tl.float32)
            tl.store(Y + (rindex + N * row_idx), xs * rsqrt, mask)


def _get_mblock(feature_dim: int) -> int:
    return max(1, min(DEFAULT_MBLOCK, MAX_BLOCK_ELEMENTS // feature_dim))


def _get_split_n_tiling() -> tuple[int, int]:
    return MAX_BLOCK_ELEMENTS // SPLIT_N_BLOCK, SPLIT_N_BLOCK


def l2norm_fwd(x: torch.Tensor, eps: float = 1e-6, output_dtype: torch.dtype | None = None):
    x_shape_og = x.shape
    x = x.reshape(-1, x.shape[-1])
    # allocate output
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)
    assert y.stride(-1) == 1
    T, D = x.shape[0], x.shape[-1]
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError(f"l2norm_fwd: This layer doesn't support feature dim >= 64KB, got {D}.")

    num_core = get_vectorcore_num()
    if D <= MAX_BLOCK_ELEMENTS:
        MBLOCK = _get_mblock(D)
        main_bs = triton.cdiv(T, num_core)
        num_sub_blocks = triton.cdiv(main_bs, MBLOCK)
        grid = (num_core,)
        l2norm_fwd_kernel2_loop[grid](
            X=x,
            Y=y,
            eps=eps,
            M=T,
            N=D,
            MBLOCK=MBLOCK,
            NUM_CHUNKS=num_sub_blocks,
        )
    else:
        MBLOCK, NBLOCK = _get_split_n_tiling()
        main_bs = triton.cdiv(T, num_core)
        num_sub_blocks = triton.cdiv(main_bs, MBLOCK)
        num_n_chunks = triton.cdiv(D, NBLOCK)
        grid = (num_core,)
        l2norm_fwd_kernel2_loop_split_n[grid](
            X=x,
            Y=y,
            eps=eps,
            M=T,
            N=D,
            MBLOCK=MBLOCK,
            NBLOCK=NBLOCK,
            NUM_N_CHUNKS=num_n_chunks,
            NUM_CHUNKS=num_sub_blocks,
        )

    return y.view(x_shape_og)
