# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Keep the shared KV service alive for the entire local mp engine lifetime."""

from __future__ import annotations

import weakref
from multiprocessing import connection

import vllm.v1.engine.utils as engine_utils
from vllm.utils.system_utils import get_mp_context

from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.shared.bootstrap import (
    instance_identity,
    service_settings,
    shared_options,
    validate_shared_config,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.shared.service import SharedService

_ORIGINAL_MANAGER = engine_utils.CoreEngineProcManager


def _finalize_shared_service(engine_finalizer, service):
    engine_finalizer()
    service.close()


class SharedKVCoreEngineProcManager(_ORIGINAL_MANAGER):
    _ascend_shared_cache_manager = True

    def __init__(self, local_engine_count, start_index, local_start_index, vllm_config, *args, **kwargs):
        self._shared_service = None
        self._shared_finalizer = None
        extra = shared_options(vllm_config)
        if extra is not None:
            validate_shared_config(vllm_config)
            if start_index != 0 or local_engine_count != vllm_config.parallel_config.data_parallel_size:
                raise ValueError("Shared KV cache must own every local DP in a single service process")
            identity = instance_identity(vllm_config)
            service = SharedService(service_settings(vllm_config), get_mp_context())
            self._shared_service = service
            extra["_cross_dp_connection"] = service.connection
            extra["_cross_dp_identity"] = identity
        try:
            super().__init__(local_engine_count, start_index, local_start_index, vllm_config, *args, **kwargs)
        except BaseException:
            if self._shared_service is not None:
                self._shared_service.close()
            raise
        if self._shared_service is not None:
            self._shared_finalizer = weakref.finalize(
                self, _finalize_shared_service, self._finalizer, self._shared_service
            )

    def shutdown(self, timeout=None):
        # Stop device workers before unmapping or reusing any shared host bytes.
        super().shutdown(timeout)
        if self._shared_finalizer is not None:
            self._shared_finalizer.detach()
        if self._shared_service is not None:
            self._shared_service.close()

    def sentinels(self):
        result = super().sentinels()
        if self._shared_service is not None:
            result.append(self._shared_service.process.sentinel)
        return result

    def finished_procs(self):
        result = super().finished_procs()
        if self._shared_service is not None:
            process = self._shared_service.process
            if process.exitcode is not None:
                result[process.name] = process.exitcode
        return result

    def monitor_engine_liveness(self):
        if self._shared_service is None:
            return super().monitor_engine_liveness()
        processes = [*self.processes, self._shared_service.process]
        by_sentinel = {process.sentinel: process for process in processes}
        while not self.manager_stopped.is_set():
            died = connection.wait(list(by_sentinel), timeout=1)
            if died:
                for sentinel in died:
                    process = by_sentinel[sentinel]
                    if process.exitcode != 0 and not self.manager_stopped.is_set():
                        self.failed_proc_name = process.name
                break
        self.shutdown()


if not getattr(_ORIGINAL_MANAGER, "_ascend_shared_cache_manager", False):
    engine_utils.CoreEngineProcManager = SharedKVCoreEngineProcManager
