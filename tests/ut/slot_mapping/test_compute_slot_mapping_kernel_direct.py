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
from vllm.v1.worker.block_table import _compute_slot_mapping_kernel

DEVICE = "npu"


def main() -> None:
    num_tokens = 4
    max_num_batched_tokens = 8192
    num_reqs = 2
    block_size = 128

    query_start_loc = torch.tensor([0, 2, 4],
                                   dtype=torch.int32,
                                   device=DEVICE)
    positions = torch.tensor([0, 17, 1, 18],
                             dtype=torch.int64,
                             device=DEVICE)
    block_table = torch.tensor(
        [
            [10, 11, 0, 0],
            [20, 21, 0, 0],
        ],
        dtype=torch.int32,
        device=DEVICE,
    )
    slot_mapping = torch.full((max_num_batched_tokens, ),
                              PAD_SLOT_ID,
                              dtype=torch.int32,
                              device=DEVICE)

    _compute_slot_mapping_kernel[(num_reqs + 1, )](
        num_tokens,
        max_num_batched_tokens,
        query_start_loc,
        positions,
        block_table,
        block_table.stride(0),
        block_size,
        slot_mapping,
        TOTAL_CP_WORLD_SIZE=1,
        TOTAL_CP_RANK=0,
        CP_KV_CACHE_INTERLEAVE_SIZE=1,
        PAD_ID=PAD_SLOT_ID,
        BLOCK_SIZE=1024,
    )
    torch.npu.synchronize()

    expected = torch.tensor([1280, 1297, 2561, 2578],
                            dtype=torch.int32,
                            device=DEVICE)
    actual = slot_mapping[:num_tokens]

    if not torch.equal(actual, expected):
        raise AssertionError(
            f"slot_mapping mismatch: actual={actual.cpu().tolist()}, "
            f"expected={expected.cpu().tolist()}")


if __name__ == "__main__":
    main()
