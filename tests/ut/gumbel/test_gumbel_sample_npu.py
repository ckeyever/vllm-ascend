# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

import torch
import torch_npu  # noqa: F401

from vllm_ascend.worker.v2.sample.gumbel import gumbel_sample

DEVICE = "npu"
VOCAB_SIZE = 32000
NUM_TOKENS_NUM_REQS_CASES = ((1, 1), (4096, 16))


def build_inputs(num_tokens: int, max_num_reqs: int):
    torch.manual_seed(2026 + num_tokens + max_num_reqs)

    logits = torch.randn(
        num_tokens,
        VOCAB_SIZE,
        dtype=torch.float32,
        device=DEVICE,
    )
    expanded_idx_mapping = torch.arange(
        num_tokens,
        dtype=torch.int32,
        device=DEVICE,
    ) % max_num_reqs
    temperature = torch.ones(
        max_num_reqs,
        dtype=torch.float32,
        device=DEVICE,
    )
    seed = torch.arange(
        1,
        max_num_reqs + 1,
        dtype=torch.int64,
        device=DEVICE,
    )
    pos = torch.arange(
        num_tokens,
        dtype=torch.int32,
        device=DEVICE,
    )
    return logits, expanded_idx_mapping, temperature, seed, pos


def torch_greedy_reference(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=-1)


def run_gumbel_case(num_tokens: int, max_num_reqs: int) -> None:
    logits, expanded_idx_mapping, temperature, seed, pos = build_inputs(
        num_tokens,
        max_num_reqs,
    )

    sampled = gumbel_sample(
        logits,
        expanded_idx_mapping,
        temperature,
        seed,
        pos,
        apply_temperature=False,
    )
    torch.npu.synchronize()

    expected = torch_greedy_reference(logits)
    if sampled.shape != (num_tokens,):
        raise AssertionError(
            f"Unexpected sampled shape: got {tuple(sampled.shape)}, "
            f"expected ({num_tokens},)."
        )
    if sampled.dtype != torch.int64:
        raise AssertionError(f"Unexpected sampled dtype: got {sampled.dtype}, expected torch.int64.")
    if not torch.equal(sampled, expected):
        mismatch = (sampled != expected).nonzero().flatten()
        first = mismatch[0].item()
        raise AssertionError(
            "gumbel_sample output does not match torch reference: "
            f"num_tokens={num_tokens}, max_num_reqs={max_num_reqs}, "
            f"mismatches={mismatch.numel()}, first_mismatch={first}, "
            f"sampled={sampled[first].item()}, expected={expected[first].item()}."
        )

    print(
        "PASS "
        f"num_tokens={num_tokens}, max_num_reqs={max_num_reqs}, vocab_size={VOCAB_SIZE}"
    )


def run_all_cases() -> None:
    for num_tokens, max_num_reqs in NUM_TOKENS_NUM_REQS_CASES:
        run_gumbel_case(num_tokens, max_num_reqs)


if __name__ == "__main__":
    run_all_cases()