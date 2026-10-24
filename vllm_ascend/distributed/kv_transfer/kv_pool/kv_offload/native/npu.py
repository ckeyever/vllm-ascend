# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Ascend specs for vLLM's native KV-cache offloading."""

from __future__ import annotations

import torch
from typing_extensions import override
from vllm.v1.kv_offload.base import CanonicalKVCaches, OffloadingWorker
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec as _CPUOffloadingSpec
from vllm.v1.kv_offload.tiering.spec import (
    TieringOffloadingSpec as _TieringOffloadingSpec,
)

from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.cpu_npu import (
    NPUOffloadingWorker,
)


class _NPUWorkerMixin:
    """Replace only the accelerator worker and upstream platform gate."""

    _worker: NPUOffloadingWorker | None

    def __init__(self, config):
        super().__init__(config)
        if self.extra_config.get("cross_dp_shared_cache", False):
            if "_cross_dp_connection" not in self.extra_config:
                raise ValueError("Shared KV offload must be started by the local mp service parent")
            # Cross-DP ownership replaces the upstream replicated rank-0 writer.
            self.replicated_layout = False

    @classmethod
    def build_metric_definitions(cls, extra_config):
        if extra_config.get("cross_dp_shared_cache", False):
            from .shared.manager import metric_definitions

            return metric_definitions()
        return super().build_metric_definitions(extra_config)

    @classmethod
    def config_info_keys(cls, extra_config):
        if extra_config.get("cross_dp_shared_cache", False):
            return ("cpu_num_chunks",)
        return super().config_info_keys(extra_config)

    def get_manager(self):
        if not self.extra_config.get("cross_dp_shared_cache", False):
            return super().get_manager()
        if self._manager is None:
            from .shared.manager import SharedOffloadingManager

            self._manager = SharedOffloadingManager(
                self.extra_config["_cross_dp_connection"],
                self.config.parallel.data_parallel_index,
                self.extra_config["_cross_dp_identity"]["hash_seed"],
            )
        return self._manager

    def create_worker(
        self,
        kv_caches: CanonicalKVCaches,
    ) -> NPUOffloadingWorker:
        raise NotImplementedError

    def get_worker(
        self,
        kv_caches: CanonicalKVCaches,
    ) -> OffloadingWorker:
        if self._worker is None:
            if self.extra_config.get("cross_dp_shared_cache", False):
                from .shared.worker import SharedNPUOffloadingWorker

                self._worker = SharedNPUOffloadingWorker(self.config, kv_caches, self.blocks_per_chunk)
            else:
                self._worker = self.create_worker(kv_caches)
        return self._worker


class NPUOffloadingSpec(_NPUWorkerMixin, _CPUOffloadingSpec):
    """Single CPU-tier offloading with pinned host tensors and NPU DMA."""

    @override
    def create_worker(
        self,
        kv_caches: CanonicalKVCaches,
    ) -> NPUOffloadingWorker:
        # Keep the single-tier path on PyTorch's pinned allocator. Unlike CUDA,
        # Ascend has no public cudaHostRegister-equivalent for an arbitrary
        # mmap buffer, and the pinned tensor path has the best proven H2D/D2H
        # performance. Consequently replicated_layout remains safely disabled
        # for this spec by the upstream _uses_shared_region() gate.
        return NPUOffloadingWorker(
            kv_caches=kv_caches,
            blocks_per_chunk=self.blocks_per_chunk,
            num_cpu_blocks=self.num_chunks,
        )


class NPUTieringOffloadingSpec(_NPUWorkerMixin, _TieringOffloadingSpec):
    """CPU primary tier plus vLLM secondary tiers on Ascend NPU."""

    def _uses_shared_region(self) -> bool:
        # Unlike the CPU-only NPU spec, tiering always connects every worker
        # and the scheduler-side primary tier through SharedOffloadRegion.
        # Advertising that fact lets upstream safely enable its single-copy
        # layout for configurations it has certified as byte-replicated
        # (currently pure MLA under the supported TP topology).
        return True

    @override
    def create_worker(
        self,
        kv_caches: CanonicalKVCaches,
    ) -> NPUOffloadingWorker:
        world_size = self.config.parallel.world_size
        if self.replicated_layout:
            rank = 0
        else:
            # engine_id identifies the DP replica; fold the process-local
            # physical device index into that replica's mmap slot range.
            rank = int(torch.npu.current_device()) % world_size

        region_kwargs = dict(num_chunks=self.num_chunks, kv_bytes_per_chunk=self.kv_bytes_per_chunk)
        worker_mmap = SharedOffloadRegion(
            engine_id=self._engine_id,
            rank=rank,
            cpu_page_size=self.cpu_page_size_per_worker,
            **region_kwargs,
        )
        return NPUOffloadingWorker(
            kv_caches=kv_caches,
            blocks_per_chunk=self.blocks_per_chunk,
            num_cpu_blocks=self.num_chunks,
            mmap_region=worker_mmap,
        )
