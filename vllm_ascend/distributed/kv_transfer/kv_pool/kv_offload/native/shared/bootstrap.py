# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Configuration for the single-node, parent-managed mp implementation."""

from __future__ import annotations


def shared_options(vllm_config):
    transfer = getattr(vllm_config, "kv_transfer_config", None)
    if transfer is None:
        return None
    extra = transfer.kv_connector_extra_config
    enabled = extra.get("cross_dp_shared_cache", False)
    if not isinstance(enabled, bool):
        raise ValueError("cross_dp_shared_cache must be a boolean")
    return extra if enabled else None


def validate_shared_config(vllm_config) -> None:
    extra = shared_options(vllm_config)
    if extra is None:
        return
    parallel = vllm_config.parallel_config
    transfer = vllm_config.kv_transfer_config
    if transfer.kv_connector != "OffloadingConnector":
        raise ValueError("cross_dp_shared_cache requires OffloadingConnector")
    if (
        parallel.data_parallel_backend != "mp"
        or parallel.nnodes != 1
        or parallel.data_parallel_size_local != parallel.data_parallel_size
        or parallel.data_parallel_external_lb
        or parallel.data_parallel_hybrid_lb
        or parallel.enable_elastic_ep
        or parallel.distributed_executor_backend not in ("mp", "uni", None)
    ):
        raise ValueError("Shared KV offload requires one parent-managed, single-node local mp deployment")
    if parallel.prefill_context_parallel_size != 1 or parallel.decode_context_parallel_size != 1:
        raise ValueError("Shared KV offload does not support context parallelism")
    if extra.get("canonical_layout", False):
        raise ValueError("Shared NPU KV offload requires canonical_layout=false")
    if extra.get("spec_name", "CPUOffloadingSpec") not in ("CPUOffloadingSpec", "TieringOffloadingSpec"):
        raise ValueError("Shared KV offload requires a Native CPU or Tiering spec")
    if extra.get("eviction_policy", "lru") != "lru" or int(extra.get("store_threshold", 0)) >= 2:
        raise ValueError("Shared KV offload currently requires LRU with store_threshold below 2")
    if int(extra.get("cpu_bytes_to_use", 0)) <= 0:
        raise ValueError("cpu_bytes_to_use must be positive for the shared instance")
    tiers = extra.get("secondary_tiers", [])
    if (
        not isinstance(tiers, list)
        or len(tiers) > 1
        or any(not isinstance(tier, dict) or tier.get("type") != "fs" for tier in tiers)
    ):
        raise ValueError("Shared KV offload supports at most one filesystem SSD tier")
    if tiers:
        if extra.get("spec_name") != "TieringOffloadingSpec" or not tiers[0].get("root_dir"):
            raise ValueError("Shared SSD offload requires TieringOffloadingSpec and an fs root_dir")
        if int(tiers[0].get("max_bytes_to_use", 0)) < 0:
            raise ValueError("Shared SSD max_bytes_to_use cannot be negative")
        if min(int(tiers[0].get("n_read_threads", 4)), int(tiers[0].get("n_write_threads", 2))) <= 0:
            raise ValueError("Shared SSD read/write thread counts must be positive")
        if tiers[0].get("module_path"):
            raise ValueError("Shared SSD offload requires the built-in filesystem tier")
    if extra.get("cache_policy_module_path"):
        raise ValueError("Shared KV offload does not support custom cache policies")


def service_settings(vllm_config) -> dict:
    extra = shared_options(vllm_config)
    assert extra is not None
    parallel = vllm_config.parallel_config
    tiers = extra.get("secondary_tiers", [])
    tier = tiers[0] if tiers else {}
    return {
        "cpu_bytes": int(extra["cpu_bytes_to_use"]),
        "dp_size": parallel.data_parallel_size,
        "world_size": parallel.world_size,
        "disk_root": tier.get("root_dir"),
        "disk_bytes": int(tier.get("max_bytes_to_use", 0)),
        "read_threads": int(tier.get("n_read_threads", 4)),
        "write_threads": int(tier.get("n_write_threads", 2)),
        "enable_events": bool(vllm_config.kv_events_config and vllm_config.kv_events_config.enable_kv_cache_events),
    }


def instance_identity(vllm_config) -> dict:
    from vllm.utils.hashing import get_hash_fn_by_name
    from vllm.v1.core.kv_cache_utils import resolve_none_hash_seed

    model = vllm_config.model_config
    cache = vllm_config.cache_config
    parallel = vllm_config.parallel_config
    hash_function = get_hash_fn_by_name(cache.prefix_caching_hash_algo)
    seed = resolve_none_hash_seed(hash_function)
    if seed != resolve_none_hash_seed(hash_function):
        raise ValueError("Shared KV offload requires a common PYTHONHASHSEED for xxhash/xxhash_cbor, or sha256")
    return {
        "hash_seed": seed,
        "model": model.model,
        "revision": getattr(model, "revision", None),
        "dtype": str(model.dtype),
        "quantization": getattr(model, "quantization", None),
        "cache_dtype": cache.cache_dtype,
        "hash_algorithm": cache.prefix_caching_hash_algo,
        "tp": parallel.tensor_parallel_size,
        "pp": parallel.pipeline_parallel_size,
        "block_size": cache.block_size,
        "chunk_options": {
            key: value
            for key, value in shared_options(vllm_config).items()
            if key in ("blocks_per_chunk", "block_size")
        },
    }
