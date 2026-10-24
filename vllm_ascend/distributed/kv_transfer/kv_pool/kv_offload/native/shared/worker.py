# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Shared mmap attachment and ownership checks around native NPU DMA."""

from __future__ import annotations

import mmap
import time

import torch

from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.cpu_npu import NPUOffloadingWorker

from .manager import SharedCPULoadStoreSpec
from .service import SharedClient

LAYOUT_START_TIMEOUT = 1200
LAYOUT_POLL_SECONDS = 0.01


class SharedRegion:
    def __init__(self, layout: dict, rank: int):
        self._layout = layout
        self._offset = layout["offsets"][rank]
        with open(layout["path"], "r+b") as stream:
            self._mmap = mmap.mmap(stream.fileno(), 0)
        self._base = torch.frombuffer(self._mmap, dtype=torch.int8)
        self._views: list[torch.Tensor] = []

    def create_next_worker_view(self, size: int):
        view = torch.as_strided(self._base, (self._layout["chunks"], size), (self._layout["stride"], 1), self._offset)
        self._offset += size
        self._views.append(view)
        return view

    def cleanup(self):
        self._views.clear()
        self._base = None
        self._mmap.close()


class SharedNPUOffloadingWorker(NPUOffloadingWorker):
    def __init__(self, config, kv_caches, blocks_per_chunk: int):
        extra = config.extra_config
        self._client = SharedClient(extra["_cross_dp_connection"])
        self._dp = config.parallel.data_parallel_index
        self._rank = config.parallel.rank
        pages = [tensor.page_size_bytes * blocks_per_chunk for tensor in kv_caches.tensors]
        schema = {
            "identity": extra["_cross_dp_identity"],
            "page_sizes": pages,
            "cache_groups": [
                (group.group_id, group.tokens_per_block, tuple(group.layer_names)) for group in config.groups
            ],
            "kv_cache_layout": config.kv_cache_layout,
            "groups": [[(ref.tensor_idx, ref.page_size_bytes) for ref in group] for group in kv_caches.group_data_refs],
            "strides": [tuple(tensor.tensor.stride()[1:]) for tensor in kv_caches.tensors],
        }
        layout = self._client.call("register_worker", self._dp, self._rank, schema)
        deadline = time.monotonic() + LAYOUT_START_TIMEOUT
        while layout is None:
            if time.monotonic() >= deadline:
                raise RuntimeError("Timed out registering all local DP/TP/PP shared KV layouts")
            time.sleep(LAYOUT_POLL_SECONDS)
            layout = self._client.call("layout")
        self._shared_jobs: dict[int, tuple[bool, list[dict]]] = {}
        super().__init__(
            kv_caches=kv_caches,
            blocks_per_chunk=blocks_per_chunk,
            num_cpu_blocks=layout["chunks"],
            mmap_region=SharedRegion(layout, self._rank),
        )

    def submit_store(self, job_id, src_spec, dst_spec):
        if not isinstance(dst_spec, SharedCPULoadStoreSpec):
            raise TypeError("Shared KV store requires generation and ownership metadata")
        self._client.call("validate_transfer", self._dp, dst_spec.descriptors, True)
        self._shared_jobs[job_id] = (True, dst_spec.descriptors)
        return super().submit_store(job_id, src_spec, dst_spec)

    def submit_load(self, job_id, src_spec, dst_spec):
        if not isinstance(src_spec, SharedCPULoadStoreSpec):
            raise TypeError("Shared KV load requires a live read lease")
        self._client.call("validate_transfer", self._dp, src_spec.descriptors, False)
        self._shared_jobs[job_id] = (False, src_spec.descriptors)
        return super().submit_load(job_id, src_spec, dst_spec)

    def get_finished(self):
        results = super().get_finished()
        stores = []
        for result in results:
            store, descriptors = self._shared_jobs.pop(result.job_id)
            if store and result.success:
                stores.extend(descriptors)
        if stores:
            self._client.call("ack_store", self._dp, self._rank, stores)
        return results
