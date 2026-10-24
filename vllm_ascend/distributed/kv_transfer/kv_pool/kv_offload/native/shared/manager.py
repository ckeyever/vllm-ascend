# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""vLLM manager adapter with step-scoped lookup tickets and DMA leases."""

from __future__ import annotations

import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Collection
from contextlib import contextmanager

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import OffloadingConnectorStats
from vllm.v1.kv_offload.base import (
    Locality,
    LookupResult,
    Medium,
    OffloadingCounterMetadata,
    OffloadingEvent,
    OffloadingGaugeMetadata,
    OffloadingManager,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.metrics.cache_hit_source import CacheHitSource

from .service import SharedClient

PENDING_LOOKUP_TIMEOUT = 30
COUNTERS = (
    "dram_hits",
    "dram_stores",
    "dram_evictions",
    "deduplicated_stores",
    "ssd_stores",
    "ssd_promotions",
    "ssd_evictions",
    "ssd_capacity_skips",
    "ssd_write_errors",
    "ssd_read_errors",
)
GAUGES = ("dram_chunks", "ssd_chunks", "read_tickets", "pending_io", "quarantined_slots")
METRIC_PREFIX = "vllm:kv_offload_shared_"


def metric_definitions() -> dict:
    definitions = {
        METRIC_PREFIX + name: OffloadingCounterMetadata(documentation=f"Instance-wide shared KV {name}.")
        for name in COUNTERS
    }
    definitions.update(
        {
            METRIC_PREFIX + name: OffloadingGaugeMetadata(documentation=f"Instance-wide shared KV {name}.")
            for name in GAUGES
        }
    )
    return definitions


class SharedCPULoadStoreSpec(CPULoadStoreSpec):
    def __init__(self, descriptors: list[dict]):
        super().__init__([descriptor["slot"] for descriptor in descriptors])
        self.descriptors = descriptors


class SharedOffloadingManager(OffloadingManager):
    def __init__(self, connection: dict, dp_rank: int, expected_seed: str) -> None:
        self.client = SharedClient(connection)
        self.dp_rank = dp_rank
        self._expected_seed = expected_seed
        self._mutex = threading.Lock()
        self._scopes: dict[str, str] = {}
        self._results: dict[tuple[str, bytes], dict] = {}
        self._pending: dict[tuple[str, bytes], float] = {}
        self._disk_sources: set[tuple[str, bytes]] = set()
        self._stores: dict[tuple[str, bytes], dict] = {}
        self._loads: dict[tuple[str, bytes], deque[dict]] = defaultdict(deque)
        self._stats_snapshot: dict[str, int] = {}

    @property
    @contextmanager
    def lock(self):
        # Local state is exclusive; remote entries stay pinned by tickets.
        with self._mutex:
            try:
                yield
            finally:
                for scope in self._scopes.values():
                    self.client.call("release_scope", self.dp_rank, scope)
                self._scopes.clear()
                self._results.clear()

    def prefetch(self, keys: Collection[bytes], req_context: ReqContext) -> None:
        request_id = req_context.req_id
        missing = list(dict.fromkeys(key for key in keys if (request_id, key) not in self._results))
        if not missing:
            return
        scope = self._scopes.setdefault(request_id, uuid.uuid4().hex)
        allow_disk = req_context.load_tier_filter.allows(Medium.STORAGE, Locality.LOCAL)
        results = self.client.call("lookup", self.dp_rank, scope, missing, allow_disk)
        for key, result in zip(missing, results):
            identity = (request_id, key)
            self._results[identity] = result
            if result.get("source") == "disk":
                self._disk_sources.add(identity)

    def lookup(self, key, req_context: ReqContext) -> LookupResult:
        identity = (req_context.req_id, key)
        self.prefetch([key], req_context)
        result = self._results[identity]
        if result["status"] == "hit":
            self._pending.pop(identity, None)
            return LookupResult.HIT
        if result["status"] == "pending":
            started = self._pending.setdefault(identity, time.monotonic())
            if time.monotonic() - started < PENDING_LOOKUP_TIMEOUT:
                return LookupResult.HIT_PENDING
        return LookupResult.MISS

    def get_load_source(self, key, req_context: ReqContext) -> CacheHitSource:
        return CacheHitSource.DISK if (req_context.req_id, key) in self._disk_sources else CacheHitSource.HOST

    def prepare_load(self, keys: Collection[bytes], req_context: ReqContext):
        keys = list(keys)
        descriptors = self.client.call("prepare_load", self.dp_rank, self._scopes[req_context.req_id], keys)
        for key, descriptor in zip(keys, descriptors):
            self._loads[req_context.req_id, key].append(descriptor)
            self._results.pop((req_context.req_id, key), None)
        return SharedCPULoadStoreSpec(descriptors)

    def complete_load(self, keys: Collection[bytes], req_context: ReqContext):
        leases = []
        for key in keys:
            identity = (req_context.req_id, key)
            descriptors = self._loads[identity]
            leases.append(descriptors.popleft()["lease"])
            if not descriptors:
                del self._loads[identity]
        self.client.call("release_load", leases)

    def prepare_store(self, keys: Collection[bytes], req_context: ReqContext):
        keys = list(dict.fromkeys(keys))
        operation = uuid.uuid4().hex
        result = self.client.call("reserve_store", self.dp_rank, operation, keys)
        self.client.call("forget_operation", self.dp_rank, operation)
        descriptors = result["writes"]
        if not descriptors and "capacity" in result["statuses"]:
            return None
        for descriptor in descriptors:
            self._stores[req_context.req_id, descriptor["key"]] = descriptor
        return PrepareStoreOutput(
            keys_to_store=[descriptor["key"] for descriptor in descriptors],
            store_spec=SharedCPULoadStoreSpec(descriptors),
            evicted_keys=[],
        )

    def complete_store(self, keys: Collection[bytes], req_context: ReqContext, success: bool = True):
        descriptors = [self._stores.pop((req_context.req_id, key)) for key in keys]
        self.client.call("commit_store" if success else "abort_store", self.dp_rank, descriptors)

    def touch(self, keys: Collection[bytes], req_context: ReqContext):
        self.client.call("touch", list(keys))

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        from vllm.v1.core.kv_cache_utils import get_none_hash_seed

        if get_none_hash_seed() != self._expected_seed:
            raise ValueError("DP prefix hash seed differs from the shared instance")
        return RequestOffloadingContext()

    def take_events(self):
        return [
            OffloadingEvent(keys=item["keys"], medium=Medium(item["medium"]), removed=item["removed"])
            for item in self.client.call("take_events", self.dp_rank)
        ]

    def on_request_finished(self, req_context: ReqContext) -> None:
        # Completion callbacks can follow request_finished: keep load/store leases.
        self._pending = {key: value for key, value in self._pending.items() if key[0] != req_context.req_id}
        self._disk_sources = {key for key in self._disk_sources if key[0] != req_context.req_id}

    def reset_cache(self) -> None:
        self.client.call("reset")
        self._pending.clear()
        self._stores.clear()
        self._loads.clear()
        self._results.clear()
        self._disk_sources.clear()

    def config_info(self):
        layout = self.client.call("layout")
        return [{"cpu_num_chunks": layout["chunks"]}]

    def get_stats(self):
        # One physical pool: emitting every DP's copy would multiply counters.
        if self.dp_rank != 0:
            return None
        current = self.client.call("stats")
        stats = OffloadingConnectorStats()
        for name in COUNTERS:
            stats.increase_counter(METRIC_PREFIX + name, current.get(name, 0) - self._stats_snapshot.get(name, 0))
        for name in GAUGES:
            stats.set_gauge(METRIC_PREFIX + name, current.get(name, 0))
        self._stats_snapshot = current
        return stats

    def shutdown(self) -> None:
        for scope in self._scopes.values():
            self.client.call("release_scope", self.dp_rank, scope)
        self._scopes.clear()
        # Physical buffers and submitted DMA leases belong to the parent service.
