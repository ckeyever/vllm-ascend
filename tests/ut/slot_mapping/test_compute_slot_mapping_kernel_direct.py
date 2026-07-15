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
# This file is a part of the vllm-ascend project.
#

import torch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
#from vllm.v1.worker.block_table import _compute_slot_mapping_kernel
from vllm_ascend.worker.block_table import _compute_slot_mapping_kernel
from vllm.utils.math_utils import cdiv

DEVICE = "npu"
BLOCK_SIZE = 128
# A typical online-serving shape: 256 concurrent requests and 32K context
# length with 128-token KV blocks.
MAX_NUM_REQS = 256
MAX_NUM_BLOCKS_PER_REQ = 256


def _make_block_table() -> torch.Tensor:
    return torch.arange(MAX_NUM_REQS * MAX_NUM_BLOCKS_PER_REQ,
                        dtype=torch.int32,
                        device=DEVICE).reshape(MAX_NUM_REQS,
                                               MAX_NUM_BLOCKS_PER_REQ)


def _expected_slot_mapping(
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    query_start_loc_cpu = query_start_loc.cpu()
    positions_cpu = positions.cpu()
    num_reqs = query_start_loc_cpu.numel() - 1
    req_indices = torch.repeat_interleave(
        torch.arange(num_reqs, dtype=torch.int64),
        query_start_loc_cpu[1:] - query_start_loc_cpu[:-1],
    )
    block_indices = positions_cpu // BLOCK_SIZE
    block_offsets = positions_cpu % BLOCK_SIZE
    block_numbers = req_indices * MAX_NUM_BLOCKS_PER_REQ + block_indices
    return (block_numbers * BLOCK_SIZE + block_offsets).to(torch.int32)


def _assert_equal(case_name: str, actual: torch.Tensor,
                  expected: torch.Tensor) -> None:
    if torch.equal(actual, expected):
        return

    mismatch_indices = torch.nonzero(actual != expected).flatten()
    first_mismatch = int(mismatch_indices[0].item())
    raise AssertionError(
        f"{case_name} slot_mapping mismatch: "
        f"num_mismatches={mismatch_indices.numel()}, "
        f"first_mismatch={first_mismatch}, "
        f"actual={int(actual[first_mismatch].item())}, "
        f"expected={int(expected[first_mismatch].item())}")


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()

def _run_case(
    case_name: str,
    num_tokens: int,
    max_num_batched_tokens: int,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    num_reqs = query_start_loc.numel() - 1
    block_table = _make_block_table()
    slot_mapping = torch.full((max_num_batched_tokens, ),
                              PAD_SLOT_ID,
                              dtype=torch.int32,
                              device=DEVICE)

    num_pad_tokens = max(max_num_batched_tokens - num_tokens, 0)
    num_pad_blocks = cdiv(num_pad_tokens, 4096)
    grid = (num_reqs + 1,)
    grid = (num_reqs + num_pad_blocks,)
    _compute_slot_mapping_kernel[grid](
        num_tokens,
        max_num_batched_tokens,
        num_reqs,
        query_start_loc,
        positions,
        block_table,
        block_table.stride(0),
        BLOCK_SIZE,
        slot_mapping,
        BLOCK_SIZE,
        1,
        TOTAL_CP_WORLD_SIZE=1,
        TOTAL_CP_RANK=0,
        CP_KV_CACHE_INTERLEAVE_SIZE=1,
        PAD_ID=PAD_SLOT_ID,
        BLOCK_SIZE=1024,
        BLOCK_TABLE_WINDOW_SIZE=_next_power_of_2(cdiv(1024, BLOCK_SIZE) + 1)
    )
    torch.npu.synchronize()

    actual = slot_mapping[:num_tokens].cpu()
    expected = _expected_slot_mapping(query_start_loc, positions)
    _assert_equal(case_name, actual, expected)

    tail = slot_mapping[num_tokens:max_num_batched_tokens].cpu()
    if tail.numel() == 0:
        return
    expected_tail = torch.full_like(tail, PAD_SLOT_ID)
    _assert_equal(f"{case_name} tail", tail, expected_tail)


def _case_small_num_tokens_large_max_num_batched_tokens() -> None:
    _run_case(
        case_name="small_num_tokens_large_max_num_batched_tokens",
        num_tokens=4,
        max_num_batched_tokens=8192,
        query_start_loc=torch.tensor([0, 2, 4],
                                     dtype=torch.int32,
                                     device=DEVICE),
        positions=torch.tensor([0, 17, 1, 18],
                               dtype=torch.int64,
                               device=DEVICE),
    )


def _case_normal_num_tokens_and_max_num_batched_tokens() -> None:
    tokens_per_req = 32
    num_tokens = MAX_NUM_REQS * tokens_per_req
    query_start_loc = torch.arange(0,
                                   num_tokens + 1,
                                   tokens_per_req,
                                   dtype=torch.int32,
                                   device=DEVICE)
    positions = torch.arange(num_tokens, dtype=torch.int64,
                             device=DEVICE) % tokens_per_req

    _run_case(
        case_name="normal_num_tokens_and_max_num_batched_tokens",
        num_tokens=num_tokens,
        max_num_batched_tokens=num_tokens,
        query_start_loc=query_start_loc,
        positions=positions,
    )


def main() -> None:
    _case_small_num_tokens_large_max_num_batched_tokens()
    _case_normal_num_tokens_and_max_num_batched_tokens()


if __name__ == "__main__":
    main()