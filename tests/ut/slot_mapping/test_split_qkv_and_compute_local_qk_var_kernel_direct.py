#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch

from vllm_ascend.ops.triton.linearnorm.split_qkv_tp_rmsnorm_rope import (
    _split_qkv_and_compute_local_qk_var_kernel,
)
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton, get_vectorcore_num

DEVICE = "npu"
DTYPE = torch.bfloat16
INPUT_PADDING_SENTINEL = 25.0
OUTPUT_SENTINEL = -7.0
VAR_SENTINEL = -1.0
VAR_ATOL = 1e-5
VAR_RTOL = 1e-5


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _assert_equal(
    case_name: str,
    tensor_name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    try:
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    except AssertionError as error:
        raise AssertionError(
            f"{case_name} {tensor_name} mismatch: {error}"
        ) from error


def _assert_var_close(
    case_name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    try:
        torch.testing.assert_close(
            actual,
            expected,
            atol=VAR_ATOL,
            rtol=VAR_RTOL,
        )
    except AssertionError as error:
        raise AssertionError(f"{case_name} qk_var mismatch: {error}") from error


@torch.inference_mode()
def _run_case(
    case_name: str,
    num_tokens: int,
    output_capacity: int,
    num_programs: int,
    q_cols: int,
    k_cols: int,
    input_padding: int,
    seed: int,
) -> None:
    generator = torch.Generator().manual_seed(seed)
    qkv_cols = q_cols + 2 * k_cols
    qkv_stride = qkv_cols + input_padding

    qkv_cpu = torch.full(
        (num_tokens, qkv_stride),
        INPUT_PADDING_SENTINEL,
        dtype=DTYPE,
    )
    qkv_cpu[:, :qkv_cols] = torch.randn(
        num_tokens,
        qkv_cols,
        generator=generator,
        dtype=torch.float32,
    ).to(DTYPE)

    expected_q = qkv_cpu[:, :q_cols].contiguous()
    expected_k = qkv_cpu[:, q_cols : q_cols + k_cols].contiguous()
    expected_v = qkv_cpu[:, q_cols + k_cols : qkv_cols].contiguous()
    expected_q_var = (
        expected_q.to(torch.float32).pow(2).sum(dim=-1) * (1.0 / q_cols)
    )
    expected_k_var = (
        expected_k.to(torch.float32).pow(2).sum(dim=-1) * (1.0 / k_cols)
    )
    expected_qk_var = torch.stack([expected_q_var, expected_k_var], dim=-1)

    qkv = qkv_cpu.to(DEVICE)
    q = torch.full(
        (output_capacity, q_cols),
        OUTPUT_SENTINEL,
        dtype=DTYPE,
        device=DEVICE,
    )
    k = torch.full(
        (output_capacity, k_cols),
        OUTPUT_SENTINEL,
        dtype=DTYPE,
        device=DEVICE,
    )
    v = torch.full_like(k, OUTPUT_SENTINEL)
    qk_var = torch.full(
        (output_capacity, 2),
        VAR_SENTINEL,
        dtype=torch.float32,
        device=DEVICE,
    )

    num_vectorcore = get_vectorcore_num()
    grid = (min(num_tokens, num_vectorcore),)
    _split_qkv_and_compute_local_qk_var_kernel[grid](
        qkv,
        q,
        k,
        v,
        qk_var,
        num_tokens,
        q_cols,
        k_cols,
        _next_power_of_2(q_cols),
        _next_power_of_2(k_cols),
        qkv_stride,
        1.0 / q_cols,
        1.0 / k_cols,
    )
    torch.npu.synchronize()

    actual_q = q.cpu()
    actual_k = k.cpu()
    actual_v = v.cpu()
    actual_qk_var = qk_var.cpu()
    _assert_equal(case_name, "q", actual_q[:num_tokens], expected_q)
    _assert_equal(case_name, "k", actual_k[:num_tokens], expected_k)
    _assert_equal(case_name, "v", actual_v[:num_tokens], expected_v)
    _assert_var_close(
        case_name,
        actual_qk_var[:num_tokens],
        expected_qk_var,
    )

    expected_q_tail = torch.full_like(actual_q[num_tokens:], OUTPUT_SENTINEL)
    expected_k_tail = torch.full_like(actual_k[num_tokens:], OUTPUT_SENTINEL)
    expected_v_tail = torch.full_like(actual_v[num_tokens:], OUTPUT_SENTINEL)
    expected_var_tail = torch.full_like(actual_qk_var[num_tokens:], VAR_SENTINEL)
    _assert_equal(case_name, "q tail", actual_q[num_tokens:], expected_q_tail)
    _assert_equal(case_name, "k tail", actual_k[num_tokens:], expected_k_tail)
    _assert_equal(case_name, "v tail", actual_v[num_tokens:], expected_v_tail)
    _assert_equal(
        case_name,
        "qk_var tail",
        actual_qk_var[num_tokens:],
        expected_var_tail,
    )


def _case_single_token_non_power_of_two_columns() -> None:
    _run_case(
        case_name="single_token_non_power_of_two_columns",
        num_tokens=1,
        output_capacity=4,
        num_programs=1,
        q_cols=96,
        k_cols=40,
        input_padding=7,
        seed=0,
    )


def _case_grid_stride_loop_and_masked_tail() -> None:
    _run_case(
        case_name="grid_stride_loop_and_masked_tail",
        num_tokens=19,
        output_capacity=24,
        num_programs=3,
        q_cols=128,
        k_cols=64,
        input_padding=0,
        seed=1,
    )


def main() -> None:
    init_device_properties_triton()
    _case_single_token_non_power_of_two_columns()
    _case_grid_stride_loop_and_masked_tail()
    print("OK...................OK")


if __name__ == "__main__":
    main()
