# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Batch the candidate keys selected by upstream's prefix/group matching."""

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import OffloadingConnectorScheduler


class SharedOffloadingScheduler(OffloadingConnectorScheduler):
    def _maximal_prefix_lookup(self, keys, req_context, req, group_config, start_chunk_idx):
        keys = list(keys)
        self.manager.prefetch(keys, req_context)
        return super()._maximal_prefix_lookup(keys, req_context, req, group_config, start_chunk_idx)

    def _sliding_window_lookup(self, keys, sliding_window_size, req_context, initial_window_size=None):
        self.manager.prefetch(keys, req_context)
        return super()._sliding_window_lookup(keys, sliding_window_size, req_context, initial_window_size)
