# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Native adapter contracts with actual vLLM descriptors and CPU mmap views."""

import mmap
import pickle

import pytest
from vllm.v1.core.kv_cache_utils import get_none_hash_seed
from vllm.v1.kv_offload.base import ReqContext

from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.shared.manager import (
    SharedCPULoadStoreSpec,
    SharedOffloadingManager,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.shared.service import SharedClient, SharedService
from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.shared.worker import SharedRegion


@pytest.fixture
def service():
    owner = SharedService({"cpu_bytes": 2 * mmap.PAGESIZE, "dp_size": 2, "world_size": 1})
    client = SharedClient(owner.connection)
    schema = {"identity": {"model": "test"}, "page_sizes": [16], "groups": [[0]]}
    client.call("register_worker", 0, 0, schema)
    client.call("register_worker", 1, 0, schema)
    try:
        yield owner, client
    finally:
        owner.close()


def test_shared_transfer_metadata_keeps_generation_and_lease_after_pickle():
    descriptor = {
        "pool_epoch": "instance",
        "key": b"prefix",
        "slot": 2,
        "generation": 9,
        "token": "owner",
        "lease": "reader",
    }
    decoded = pickle.loads(pickle.dumps(SharedCPULoadStoreSpec([descriptor])))
    assert decoded.block_ids.tolist() == [2]
    assert decoded.descriptors == [descriptor]


def test_two_requests_in_one_step_have_independent_read_leases(service):
    owner, client = service
    writer = SharedOffloadingManager(owner.connection, 0, get_none_hash_seed())
    reader = SharedOffloadingManager(owner.connection, 1, get_none_hash_seed())
    request = ReqContext("source")
    with writer.lock:
        store = writer.prepare_store([b"key"], request)
        client.call("ack_store", 0, 0, store.store_spec.descriptors)
        writer.complete_store([b"key"], request)
    requests = [ReqContext("reader0"), ReqContext("reader1")]
    with reader.lock:
        for context in requests:
            assert reader.lookup(b"key", context).name == "HIT"
        loads = [reader.prepare_load([b"key"], context) for context in requests]
    assert loads[0].descriptors[0]["lease"] != loads[1].descriptors[0]["lease"]
    assert client.call("stats")["read_tickets"] == 2
    for context in requests:
        with reader.lock:
            reader.complete_load([b"key"], context)
    assert client.call("stats")["read_tickets"] == 0


def test_request_finished_keeps_inflight_store_until_completion(service):
    owner, client = service
    manager = SharedOffloadingManager(owner.connection, 0, get_none_hash_seed())
    context = ReqContext("finished")
    with manager.lock:
        store = manager.prepare_store([b"key"], context)
        manager.on_request_finished(context)
    client.call("ack_store", 0, 0, store.store_spec.descriptors)
    with manager.lock:
        manager.complete_store([b"key"], context)
    assert client.call("stats")["dram_stores"] == 1


def test_cancelled_lookup_scope_releases_pin_without_starting_dma(service):
    owner, client = service
    manager = SharedOffloadingManager(owner.connection, 0, get_none_hash_seed())
    context = ReqContext("source")
    with manager.lock:
        store = manager.prepare_store([b"key"], context)
        client.call("ack_store", 0, 0, store.store_spec.descriptors)
        manager.complete_store([b"key"], context)
    with manager.lock:
        assert manager.lookup(b"key", ReqContext("cancelled")).name == "HIT"
        assert client.call("stats")["read_tickets"] == 1
    assert client.call("stats")["read_tickets"] == 0


def test_worker_view_points_at_registered_rank_slice_without_copy(tmp_path):
    path = tmp_path / "kv.mmap"
    path.write_bytes(bytes(2 * mmap.PAGESIZE))
    layout = {"path": str(path), "chunks": 2, "stride": mmap.PAGESIZE, "offsets": [0, 64]}
    region = SharedRegion(layout, 1)
    view = region.create_next_worker_view(16)
    view[1].fill_(23)
    assert path.read_bytes()[mmap.PAGESIZE + 64 : mmap.PAGESIZE + 80] == bytes([23]) * 16
    del view
    region.cleanup()
