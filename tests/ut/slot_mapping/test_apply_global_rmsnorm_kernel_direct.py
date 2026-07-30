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
    _apply_global_rmsnorm_kernel,
)
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

DEVICE = "npu"
DTYPE = torch.bfloat16
EPS = 1e-6
DEFAULT_ATOL = 5e-2
DEFAULT_RTOL = 5e-3


def _apply_rope_neox(
    values: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> torch.Tensor:
    half = rotary_dim // 2
    values_f32 = values.to(torch.float32)
    cos_f32 = cos[:, None, :half].to(torch.float32)
    sin_f32 = sin[:, None, :half].to(torch.float32)

    first_half = values_f32[..., :half]
    second_half = values_f32[..., half:rotary_dim]
    rotated = torch.cat(
        [
            first_half * cos_f32 - second_half * sin_f32,
            second_half * cos_f32 + first_half * sin_f32,
        ],
        dim=-1,
    )
    return torch.cat([rotated, values_f32[..., rotary_dim:]], dim=-1).to(
        values.dtype
    )


def _reference(
    values: torch.Tensor,
    weight: torch.Tensor,
    global_var: torch.Tensor,
    tp_world: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> torch.Tensor:
    normalized = (
        values.to(torch.float32)
        * torch.rsqrt(global_var[:, None, None] / tp_world + EPS)
        * weight.to(torch.float32)[None, :, :]
    ).to(values.dtype)
    return _apply_rope_neox(normalized, cos, sin, rotary_dim)


def _assert_close(
    case_name: str,
    tensor_name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    try:
        torch.testing.assert_close(
            actual.to(torch.float32),
            expected.to(torch.float32),
            atol=DEFAULT_ATOL,
            rtol=DEFAULT_RTOL,
        )
    except AssertionError as error:
        raise AssertionError(f"{case_name} {tensor_name} mismatch: {error}") from error


@torch.inference_mode()
def _run_case(
    case_name: str,
    num_tokens: int,
    num_programs: int,
    q_num_heads: int,
    k_num_heads: int,
    head_dim: int,
    rotary_dim: int,
    tp_world: int,
    cos_padding: int,
    seed: int,
) -> None:
    generator = torch.Generator().manual_seed(seed)
    half = rotary_dim // 2
    q_cols = q_num_heads * head_dim
    k_cols = k_num_heads * head_dim

    q_cpu = torch.randn(
        num_tokens, q_num_heads, head_dim, generator=generator, dtype=torch.float32
    ).to(DTYPE)
    k_cpu = torch.randn(
        num_tokens, k_num_heads, head_dim, generator=generator, dtype=torch.float32
    ).to(DTYPE)
    q_weight_cpu = (
        torch.randn(q_num_heads, head_dim, generator=generator, dtype=torch.float32)
        * 0.1
        + 1.0
    )
    k_weight_cpu = (
        torch.randn(k_num_heads, head_dim, generator=generator, dtype=torch.float32)
        * 0.1
        + 1.0
    )

    cs_row_stride = half + cos_padding
    cos_cpu = torch.zeros(num_tokens, cs_row_stride, dtype=DTYPE)
    sin_cpu = torch.zeros_like(cos_cpu)
    angles = torch.randn(
        num_tokens, half, generator=generator, dtype=torch.float32
    )
    cos_cpu[:, :half] = torch.cos(angles).to(DTYPE)
    sin_cpu[:, :half] = torch.sin(angles).to(DTYPE)

    q_local_var = q_cpu.to(torch.float32).pow(2).mean(dim=(1, 2))
    k_local_var = k_cpu.to(torch.float32).pow(2).mean(dim=(1, 2))
    remote_q_var = torch.linspace(0.5, 1.5, num_tokens, dtype=torch.float32)
    remote_k_var = torch.linspace(1.5, 0.5, num_tokens, dtype=torch.float32)
    q_global_var = q_local_var + (tp_world - 1) * remote_q_var
    k_global_var = k_local_var + (tp_world - 1) * remote_k_var
    qk_global_var_cpu = torch.stack([q_global_var, k_global_var], dim=-1)

    expected_q = _reference(
        q_cpu,
        q_weight_cpu,
        q_global_var,
        tp_world,
        cos_cpu,
        sin_cpu,
        rotary_dim,
    )
    expected_k = _reference(
        k_cpu,
        k_weight_cpu,
        k_global_var,
        tp_world,
        cos_cpu,
        sin_cpu,
        rotary_dim,
    )

    q = q_cpu.reshape(num_tokens, q_cols).to(DEVICE)
    k = k_cpu.reshape(num_tokens, k_cols).to(DEVICE)
    cos = cos_cpu.to(DEVICE)
    sin = sin_cpu.to(DEVICE)
    q_weight = q_weight_cpu.to(DEVICE)
    k_weight = k_weight_cpu.to(DEVICE)
    qk_global_var = qk_global_var_cpu.to(DEVICE)
    num_vectorcore = get_vectorcore_num()
    grid = (min(num_tokens, num_vectorcore),)

    _apply_global_rmsnorm_kernel[(grid,)](
        q,
        k,
        cos,
        sin,
        cs_row_stride,
        q_weight,
        k_weight,
        qk_global_var,
        EPS,
        1.0 / tp_world,
        num_tokens,
        q_cols,
        k_cols,
        q_num_heads,
        k_num_heads,
        head_dim,
        rotary_dim,
        half,
    )
    torch.npu.synchronize()

    actual_q = q.view(num_tokens, q_num_heads, head_dim).cpu()
    actual_k = k.view(num_tokens, k_num_heads, head_dim).cpu()
    _assert_close(case_name, "q", actual_q, expected_q)
    _assert_close(case_name, "k", actual_k, expected_k)
    print("OK...................OK")


def _case_single_token_full_rotary_dim() -> None:
    _run_case(
        case_name="single_token_full_rotary_dim",
        num_tokens=1,
        num_programs=1,
        q_num_heads=2,
        k_num_heads=1,
        head_dim=128,
        rotary_dim=128,
        tp_world=1,
        cos_padding=0,
        seed=0,
    )


def _case_multi_program_partial_rotary_dim_and_tp() -> None:
    _run_case(
        case_name="multi_program_partial_rotary_dim_and_tp",
        num_tokens=7,
        num_programs=3,
        q_num_heads=8,
        k_num_heads=2,
        head_dim=128,
        rotary_dim=64,
        tp_world=4,
        cos_padding=5,
        seed=1,
    )


def main() -> None:
    _case_single_token_full_rotary_dim()
    _case_multi_program_partial_rotary_dim_and_tp()


if __name__ == "__main__":
    main()
