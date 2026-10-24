# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Exercise the real mp lifecycle patch with only process/device boundaries mocked."""

import gc
import importlib.util
import sys
import threading
import types
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
NATIVE_DIR = ROOT / "vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/native"
sys.path.insert(0, str(NATIVE_DIR))
from shared import bootstrap, service  # noqa: E402


def _modules(names):
    result = {}
    for name in names:
        parts = name.split(".")
        for length in range(1, len(parts) + 1):
            prefix = ".".join(parts[:length])
            if prefix not in result:
                result[prefix] = types.ModuleType(prefix)
                result[prefix].__path__ = []
                if length > 1:
                    setattr(result[".".join(parts[: length - 1])], parts[length - 1], result[prefix])
    return result


class SharedLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.events = []
        events = self.events

        class BaseManager:
            def __init__(self, local_count, start, local_start, config, **kwargs):
                self.processes = [SimpleNamespace(sentinel=1, name="EngineCore", exitcode=None)]
                self.manager_stopped = threading.Event()
                self.failed_proc_name = None
                self._finalizer = weakref.finalize(self, events.append, "engine-finalized")
                self.copied_connection = config.kv_transfer_config.kv_connector_extra_config.get("_cross_dp_connection")
                events.append("engine-start")
                if kwargs.get("fail"):
                    raise ValueError("engine startup failed")

            def shutdown(self, timeout=None):
                events.append("engine-stop")
                self.manager_stopped.set()
                self._finalizer.detach()

            def sentinels(self):
                return [1]

            def finished_procs(self):
                return {}

            def monitor_engine_liveness(self):
                events.append("original-monitor")

        class Service:
            def __init__(self, settings, context):
                events.append("cache-start")
                self.connection = {"address": "local-test", "authkey": "test"}
                self.process = SimpleNamespace(sentinel=2, name="SharedKV", exitcode=None)
                self.closed = False

            def close(self):
                if not self.closed:
                    events.append("cache-stop")
                    self.closed = True

        canonical = "vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.shared"
        modules = _modules(
            ["vllm.v1.engine.utils", "vllm.utils.system_utils", canonical + ".bootstrap", canonical + ".service"]
        )
        modules["vllm.v1.engine.utils"].CoreEngineProcManager = BaseManager
        modules["vllm.utils.system_utils"].get_mp_context = lambda: None
        modules[canonical + ".bootstrap"] = bootstrap
        modules[canonical + ".service"] = service
        self.module_patch = patch.dict(sys.modules, modules)
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        path = ROOT / "vllm_ascend/patch/platform/patch_shared_kv_offload.py"
        spec = importlib.util.spec_from_file_location("shared_mp_lifecycle_test", path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.module.SharedService = Service
        self.module.instance_identity = lambda config: {"hash_seed": "fixed"}
        self.config = SimpleNamespace(
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
                world_size=1,
            ),
            kv_events_config=None,
        )

    def manager(self, **kwargs):
        return self.module.SharedKVCoreEngineProcManager(2, 0, 0, self.config, **kwargs)

    def test_cache_starts_before_dp_config_copy_and_survives_startup(self):
        manager = self.manager()
        self.assertEqual(self.events, ["cache-start", "engine-start"])
        self.assertEqual(manager.copied_connection["address"], "local-test")
        self.assertFalse(manager._shared_service.closed)
        self.assertEqual(manager.sentinels(), [1, 2])
        manager.shutdown()
        self.assertEqual(self.events[-2:], ["engine-stop", "cache-stop"])

    def test_startup_failure_closes_owned_cache(self):
        with self.assertRaisesRegex(ValueError, "startup failed"):
            self.manager(fail=True)
        self.assertIn("cache-stop", self.events)

    def test_service_failure_participates_in_engine_liveness(self):
        manager = self.manager()
        manager._shared_service.process.exitcode = 1
        self.assertEqual(manager.finished_procs(), {"SharedKV": 1})
        with patch.object(self.module.connection, "wait", return_value=[2]):
            manager.monitor_engine_liveness()
        self.assertEqual(manager.failed_proc_name, "SharedKV")
        self.assertEqual(self.events[-2:], ["engine-stop", "cache-stop"])

    def test_disabled_feature_uses_original_manager_lifecycle(self):
        self.config.kv_transfer_config.kv_connector_extra_config.clear()
        manager = self.manager()
        self.assertEqual(self.events, ["engine-start"])
        self.assertEqual(manager.sentinels(), [1])
        manager.monitor_engine_liveness()
        self.assertEqual(self.events[-1], "original-monitor")
        manager.shutdown()

    def test_finalizer_stops_engines_before_shared_memory_owner(self):
        manager = self.manager()
        reference = weakref.ref(manager)
        del manager
        gc.collect()
        self.assertIsNone(reference())
        self.assertEqual(self.events[-2:], ["engine-finalized", "cache-stop"])


if __name__ == "__main__":
    unittest.main()
