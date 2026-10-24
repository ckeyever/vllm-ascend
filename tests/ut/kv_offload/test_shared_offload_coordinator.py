# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU-only protocol tests, including independent spawned DP processes.

Can also run directly with stdlib unittest when torch/pytest are unavailable.
"""

import json
import mmap
import multiprocessing
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# Import the dependency-free subsystem without NPU connector registration.
NATIVE_DIR = Path(__file__).resolve().parents[3] / "vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/native"
sys.path.insert(0, str(NATIVE_DIR))
from shared.bootstrap import service_settings, validate_shared_config  # noqa: E402
from shared.coordinator import SharedCache  # noqa: E402
from shared.service import SharedClient, SharedService  # noqa: E402


def _offer_from_dp(connection, dp, barrier, output):
    client = SharedClient(connection)
    barrier.wait(timeout=20)
    result = client.call("reserve_store", dp, "concurrent", [b"same-prefix"])
    output.put((dp, result))


def _read_from_dp(connection, path, output):
    client = SharedClient(connection)
    result = client.call("lookup", 1, "remote-read", [b"same-prefix"])[0]
    descriptors = client.call("prepare_load", 1, "remote-read", [b"same-prefix"])
    with open(path, "r+b") as stream, mmap.mmap(stream.fileno(), 0) as region:
        output.put((result["status"], bytes(region[:8])))
    client.call("release_load", [d["lease"] for d in descriptors])
    client.call("release_scope", 1, "remote-read")


class SharedCacheTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cache = SharedCache(self.directory.name, 2 * mmap.PAGESIZE, 2, 2)
        self.addCleanup(self.cache.close)
        schema = {"identity": {"model": "test", "hash_seed": "fixed"}, "page_sizes": [16], "groups": [[0]]}
        for dp in range(2):
            for rank in range(2):
                self.cache.register_worker(dp, rank, schema)

    def _reserve(self, key=b"a", dp=0, operation="save"):
        return self.cache.reserve_store(dp, operation, [key])["writes"]

    def _commit(self, descriptors, dp=0):
        for rank in range(2):
            self.cache.ack_store(dp, rank, descriptors)
        self.cache.commit_store(dp, descriptors)

    def _disk(self):
        self.cache.disk_dir = Path(self.directory.name) / "ssd"
        self.cache.disk_dir.mkdir()

    def test_foreign_pool_token_cannot_publish_or_load_this_pool(self):
        writes = self._reserve()
        foreign = [writes[0] | {"pool_epoch": "previous-instance"}]
        self.cache.ack_store(0, 0, foreign)
        self.cache.ack_store(0, 1, foreign)
        self.cache.commit_store(0, foreign)
        self.assertEqual(self.cache.lookup(1, "scope", [b"a"])[0]["status"], "pending")
        with self.assertRaisesRegex(RuntimeError, "Stale"):
            self.cache.validate_transfer(0, foreign, True)
        self._commit(writes)

    def test_competing_dps_get_one_writer_and_pending_is_not_readable(self):
        first = self._reserve()
        duplicate = self.cache.reserve_store(1, "other", [b"a"])
        self.assertEqual(duplicate["statuses"], ["pending"])
        self.assertEqual(duplicate["writes"], [])
        self.assertEqual(self.cache.lookup(1, "read", [b"a"])[0]["status"], "pending")
        self._commit(first)
        self.assertEqual(self.cache.lookup(1, "read", [b"a"])[0]["status"], "hit")
        self.cache.release_scope(1, "read")

    def test_retried_reservation_and_commit_are_idempotent(self):
        first = self.cache.reserve_store(0, "save", [b"a"])
        self.assertEqual(first, self.cache.reserve_store(0, "save", [b"a"]))
        self._commit(first["writes"])
        self.cache.commit_store(0, first["writes"])
        self.assertEqual(self.cache.stats()["dram_stores"], 1)

    def test_missing_tp_shard_cannot_publish_ready(self):
        writes = self._reserve()
        self.cache.ack_store(0, 0, writes)
        with self.assertRaisesRegex(RuntimeError, "every TP/PP"):
            self.cache.commit_store(0, writes)
        self.assertEqual(self.cache.lookup(1, "read", [b"a"])[0]["status"], "pending")
        self.cache.ack_store(0, 1, writes)
        self.cache.commit_store(0, writes)

    def test_lookup_to_load_transition_prevents_eviction(self):
        self._commit(self._reserve(b"a", operation="a"))
        self._commit(self._reserve(b"b", operation="b"))
        self.cache.lookup(1, "scope", [b"a"])
        self.cache.lookup(1, "scope", [b"b"])
        self.assertEqual(self.cache.reserve_store(0, "blocked", [b"c"])["statuses"], ["capacity"])
        load = self.cache.prepare_load(1, "scope", [b"a"])
        self.cache.release_scope(1, "scope")
        self.assertTrue(self._reserve(b"c", operation="c"))
        self.cache.validate_transfer(1, load, False)
        self.cache.release_load([load[0]["lease"]])
        self.cache.release_load([load[0]["lease"]])
        self.assertEqual(self.cache.stats()["read_tickets"], 0)

    def test_unconsumed_tickets_are_released_at_scope_end(self):
        self._commit(self._reserve())
        self.cache.lookup(1, "scope", [b"a", b"a"])
        self.assertEqual(self.cache.stats()["read_tickets"], 1)
        self.cache.release_scope(1, "scope")
        self.assertEqual(self.cache.stats()["read_tickets"], 0)

    def test_aborted_dma_slot_is_quarantined_and_stale_ack_cannot_publish(self):
        old = self._reserve()
        self.cache.abort_store(0, old)
        new = self._reserve(dp=1, operation="replacement")
        self.assertNotEqual(old[0]["slot"], new[0]["slot"])
        self._commit(old)
        self.assertEqual(self.cache.lookup(0, "scope", [b"a"])[0]["status"], "pending")
        self._commit(new, dp=1)
        self.assertEqual(self.cache.stats()["quarantined_slots"], 1)

    def test_ssd_only_save_is_skipped_and_promotion_is_shared(self):
        self._disk()
        first = self._reserve()
        assert self.cache._region is not None
        self.cache._region[:8] = b"payload!"
        self._commit(first)
        self.cache.drain()
        self._commit(self._reserve(b"b", operation="b"))
        self.cache.drain()
        self._commit(self._reserve(b"c", operation="c"))
        self.cache.drain()
        self.assertNotIn(b"a", self.cache._entries)
        duplicate = self.cache.reserve_store(1, "ssd-exists", [b"a"])
        self.assertEqual(duplicate["writes"], [])
        self.assertEqual(duplicate["statuses"], ["exists"])
        # Make both lookups observe the same in-flight promotion deterministically.
        with self.cache._lock:
            self.assertEqual(self.cache.lookup(0, "dp0", [b"a"])[0]["status"], "pending")
            self.assertEqual(self.cache.lookup(1, "dp1", [b"a"])[0]["status"], "pending")
            self.assertEqual(self.cache.reserve_store(1, "promotion-race", [b"a"])["writes"], [])
        self.cache.drain()
        for dp in range(2):
            self.assertEqual(self.cache.lookup(dp, "final", [b"a"])[0]["status"], "hit")
            self.cache.release_scope(dp, "final")
        self.assertEqual(self.cache.stats()["ssd_promotions"], 1)
        self.assertEqual(self.cache.stats()["ssd_stores"], 3)
        entry = self.cache._entries[b"a"]
        self.assertEqual(self.cache._region[entry.slot * mmap.PAGESIZE : entry.slot * mmap.PAGESIZE + 8], b"payload!")

    def test_ssd_corruption_is_a_miss_and_can_be_repaired(self):
        self._disk()
        self._commit(self._reserve())
        self.cache.drain()
        self._commit(self._reserve(b"b", operation="b"))
        self.cache.drain()
        self._commit(self._reserve(b"c", operation="c"))
        self.cache.drain()
        self.cache._disk_path(b"a").write_bytes(b"truncated")
        self.cache.lookup(1, "read", [b"a"])
        self.cache.drain()
        self.assertEqual(self.cache.lookup(1, "read", [b"a"])[0]["status"], "miss")
        self.assertTrue(self._reserve(b"a", operation="repair"))

    def test_failed_ssd_write_does_not_strand_pins_when_cleanup_also_fails(self):
        self._disk()
        with (
            patch("shared.coordinator.Path.open", side_effect=OSError("write failed")),
            patch("shared.coordinator.Path.unlink", side_effect=OSError("cleanup failed")),
        ):
            self._commit(self._reserve())
            self.cache.drain()
        self.assertEqual(self.cache.stats()["ssd_write_errors"], 1)
        self.assertEqual(self.cache.stats()["ssd_chunks"], 0)
        self.assertFalse(self.cache._entries[b"a"].pins)
        self.assertEqual(self.cache.lookup(1, "still-valid", [b"a"])[0]["status"], "hit")
        self.cache.release_scope(1, "still-valid")

    def test_ssd_only_lookup_can_disable_secondary_tier(self):
        self._disk()
        for key in (b"a", b"b", b"c"):
            self._commit(self._reserve(key, operation=key.hex()))
            self.cache.drain()
        self.assertEqual(self.cache.lookup(1, "filtered", [b"a"], False)[0]["status"], "miss")
        self.assertNotIn("ssd_promotions", self.cache.stats())
        self.assertEqual(self.cache.lookup(1, "filtered", [b"c"], False)[0]["status"], "hit")
        self.cache.release_scope(1, "filtered")

    def test_ssd_manifest_from_another_layout_cannot_be_loaded(self):
        self._disk()
        for key in (b"a", b"b", b"c"):
            self._commit(self._reserve(key, operation=key.hex()))
            self.cache.drain()
        manifest = self.cache._disk_path(b"a").with_suffix(".json")
        metadata = json.loads(manifest.read_text())
        metadata["schema_digest"] = "other-layout"
        manifest.write_text(json.dumps(metadata))
        self.cache.lookup(1, "invalid", [b"a"])
        self.cache.drain()
        self.assertEqual(self.cache.lookup(1, "invalid", [b"a"])[0]["status"], "miss")
        self.assertEqual(self.cache.stats()["ssd_read_errors"], 1)

    def test_ssd_capacity_is_global(self):
        self._disk()
        self.cache.disk_bytes = mmap.PAGESIZE
        self._commit(self._reserve())
        self.cache.drain()
        self._commit(self._reserve(b"b", operation="b"))
        self.cache.drain()
        self.assertEqual(self.cache.stats()["ssd_chunks"], 1)

    def test_failed_ssd_eviction_does_not_fail_a_completed_dram_store(self):
        self._disk()
        self.cache.disk_bytes = mmap.PAGESIZE
        self._commit(self._reserve())
        self.cache.drain()
        with patch("shared.coordinator.Path.unlink", side_effect=OSError("SSD unavailable")):
            self._commit(self._reserve(b"b", operation="b"))
        self.assertEqual(self.cache.lookup(1, "read", [b"b"])[0]["status"], "hit")
        self.assertEqual(self.cache.stats()["ssd_capacity_skips"], 1)
        self.assertEqual(self.cache.stats()["ssd_chunks"], 1)
        self.cache.release_scope(1, "read")

    def test_lookup_miss_then_peer_save_is_rechecked_at_store(self):
        self.assertEqual(self.cache.lookup(1, "before", [b"a"])[0]["status"], "miss")
        self._commit(self._reserve())
        self.assertEqual(self.cache.reserve_store(1, "after", [b"a"])["writes"], [])

    def test_layout_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "layout"):
            self.cache.register_worker(1, 0, {"identity": {"model": "test", "hash_seed": "fixed"}, "page_sizes": [32]})

    def test_reset_rejects_active_reads_and_keeps_slot_generations(self):
        writes = self._reserve()
        self._commit(writes)
        self.cache.lookup(1, "scope", [b"a"])
        with self.assertRaisesRegex(RuntimeError, "idle"):
            self.cache.reset()
        self.cache.release_scope(1, "scope")
        self.cache.reset()
        self.assertEqual(self.cache.stats()["dram_chunks"], 0)
        with self.assertRaisesRegex(RuntimeError, "Stale"):
            self.cache.validate_transfer(0, writes, True)


class SharedBootstrapTest(unittest.TestCase):
    def config(self):
        return SimpleNamespace(
            kv_transfer_config=SimpleNamespace(
                kv_connector="OffloadingConnector",
                kv_connector_extra_config={"cross_dp_shared_cache": True, "cpu_bytes_to_use": 8192},
            ),
            parallel_config=SimpleNamespace(
                data_parallel_backend="mp",
                nnodes=1,
                data_parallel_size_local=2,
                data_parallel_size=2,
                data_parallel_external_lb=False,
                data_parallel_hybrid_lb=False,
                enable_elastic_ep=False,
                distributed_executor_backend="mp",
                prefill_context_parallel_size=1,
                decode_context_parallel_size=1,
                world_size=2,
            ),
            kv_events_config=None,
        )

    def test_one_parent_owns_the_whole_budget(self):
        config = self.config()
        validate_shared_config(config)
        settings = service_settings(config)
        self.assertEqual(settings["cpu_bytes"], 8192)
        self.assertEqual(settings["dp_size"], 2)
        self.assertEqual(settings["world_size"], 2)

    def test_only_local_mp_configuration_is_accepted(self):
        for field, value in (
            ("data_parallel_backend", "ray"),
            ("nnodes", 2),
            ("data_parallel_size_local", 1),
            ("data_parallel_external_lb", True),
            ("distributed_executor_backend", "external_launcher"),
        ):
            with self.subTest(field=field):
                config = self.config()
                setattr(config.parallel_config, field, value)
                with self.assertRaisesRegex(ValueError, "local mp"):
                    validate_shared_config(config)

    def test_unsupported_layout_and_tiers_fail_at_configuration(self):
        for option, value in (
            ("canonical_layout", True),
            ("eviction_policy", "arc"),
            ("secondary_tiers", [{"type": "p2p"}]),
            ("secondary_tiers", [None]),
            ("cache_policy_module_path", "custom.policy"),
            ("cpu_bytes_to_use", 0),
        ):
            with self.subTest(option=option):
                config = self.config()
                config.kv_transfer_config.kv_connector_extra_config[option] = value
                with self.assertRaises(ValueError):
                    validate_shared_config(config)

    def test_default_disabled_does_not_restrict_existing_backends(self):
        config = self.config()
        config.kv_transfer_config.kv_connector_extra_config.clear()
        config.parallel_config.data_parallel_backend = "ray"
        validate_shared_config(config)

    def test_shared_ssd_uses_one_instance_quota(self):
        config = self.config()
        config.kv_transfer_config.kv_connector_extra_config.update(
            spec_name="TieringOffloadingSpec",
            secondary_tiers=[{"type": "fs", "root_dir": "/tmp/cache", "max_bytes_to_use": 65536}],
        )
        validate_shared_config(config)
        self.assertEqual(service_settings(config)["disk_bytes"], 65536)


class SharedServiceProcessTest(unittest.TestCase):
    def test_spawned_dps_deduplicate_and_read_same_shared_bytes(self):
        context = multiprocessing.get_context("spawn")
        owner = SharedService({"cpu_bytes": 2 * mmap.PAGESIZE, "dp_size": 2, "world_size": 1}, context)
        self.addCleanup(owner.close)
        client = SharedClient(owner.connection)
        schema = {"identity": {"model": "same"}, "page_sizes": [16], "groups": [[0]]}
        client.call("register_worker", 0, 0, schema)
        layout = client.call("register_worker", 1, 0, schema)
        barrier, output = context.Barrier(2), context.Queue()
        processes = [
            context.Process(target=_offer_from_dp, args=(owner.connection, dp, barrier, output)) for dp in range(2)
        ]
        for process in processes:
            process.start()
        results = [output.get(timeout=30) for _ in processes]
        for process in processes:
            process.join(30)
            self.assertEqual(process.exitcode, 0)
        writers = [(dp, result["writes"]) for dp, result in results if result["writes"]]
        self.assertEqual(len(writers), 1)
        dp, writes = writers[0]
        with open(layout["path"], "r+b") as stream, mmap.mmap(stream.fileno(), 0) as region:
            region[:8] = b"cross-DP"
        client.call("ack_store", dp, 0, writes)
        client.call("commit_store", dp, writes)
        reader = context.Process(target=_read_from_dp, args=(owner.connection, layout["path"], output))
        reader.start()
        self.assertEqual(output.get(timeout=30), ("hit", b"cross-DP"))
        reader.join(30)
        self.assertEqual(reader.exitcode, 0)
        self.assertEqual(client.call("stats")["dram_stores"], 1)
        runtime_dir = owner._directory.name
        owner.close()
        self.assertFalse(Path(runtime_dir).exists())


if __name__ == "__main__":
    unittest.main()
