# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Cross-DP Native restores through the actual single-node mp HTTP server."""

import asyncio
import json
import time

import httpx
import pytest
from prometheus_client.parser import text_string_to_metric_families

from tests.e2e.conftest import RemoteOpenAIServer

MODELS = ["Qwen/Qwen3-0.6B", "vllm-ascend/DeepSeek-V2-Lite-W8A8"]
METRIC_PREFIX = "vllm:kv_offload_shared_"


async def _metric(http, url, name):
    response = await http.get(url + "/metrics")
    response.raise_for_status()
    return sum(
        sample.value
        for family in text_string_to_metric_families(response.text)
        for sample in family.samples
        if sample.name == METRIC_PREFIX + name
    )


async def _reset_hbm(http, url):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        response = await http.post(url + "/reset_prefix_cache")
        response.raise_for_status()
        if response.json()["success"]:
            return
        await asyncio.sleep(0.1)
    raise AssertionError("HBM blocks did not become idle for prefix-cache reset")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("ssd", [False, True], ids=["shared-dram", "shared-ssd"])
@pytest.mark.asyncio
async def test_native_cross_dp_restore(model, ssd, tmp_path):
    extra = {
        "cross_dp_shared_cache": True,
        "cpu_bytes_to_use": 64 << 20,
        "blocks_per_chunk": 8,
        "spec_name": "TieringOffloadingSpec" if ssd else "CPUOffloadingSpec",
    }
    if ssd:
        extra["secondary_tiers"] = [{"type": "fs", "root_dir": str(tmp_path / "ssd"), "max_bytes_to_use": 2 << 30}]
    transfer = {"kv_connector": "OffloadingConnector", "kv_role": "kv_both", "kv_connector_extra_config": extra}
    args = [
        "--data-parallel-size",
        "2",
        "--data-parallel-backend",
        "mp",
        "--distributed-executor-backend",
        "mp",
        "--enforce-eager",
        "--trust-remote-code",
        "--max-model-len",
        "1024",
        "--block-size",
        "16",
        "--gpu-memory-utilization",
        "0.7",
        "--kv-transfer-config",
        json.dumps(transfer),
    ]
    with RemoteOpenAIServer(model, args, server_host="127.0.0.1", env_dict={"VLLM_SERVER_DEV_MODE": "1"}) as server:
        client = server.get_async_client()
        async with httpx.AsyncClient(timeout=120) as http:
            url = server.url_root

            async def generate(tokens, rank):
                return await client.completions.create(
                    model=model,
                    prompt=tokens,
                    max_tokens=1,
                    temperature=0,
                    extra_headers={"X-data-parallel-rank": str(rank)},
                )

            prompt = [101, *([100] * 512)]
            original = await generate(prompt, 0)
            await _reset_hbm(http, url)
            hits_before = await _metric(http, url, "dram_hits_total")
            restored = await generate(prompt, 1)
            assert restored.choices[0].text == original.choices[0].text
            # Drive DP0's metric output; these tiny requests have no complete chunk.
            await generate([102], 0)
            deadline = time.monotonic() + 30
            while await _metric(http, url, "dram_hits_total") <= hits_before:
                assert time.monotonic() < deadline, "Other DP did not read the shared DRAM prefix"
                await generate([103], 0)
                await asyncio.sleep(0.1)

            if ssd:
                # Distinct full prefixes exceed the shared DRAM budget, while SSD retains the original.
                for prefix in range(110, 122):
                    await generate([prefix, *([100] * 512)], 0)
                await _reset_hbm(http, url)
                promotions_before = await _metric(http, url, "ssd_promotions_total")
                from_ssd = await generate(prompt, 1)
                assert from_ssd.choices[0].text == original.choices[0].text
                await generate([104], 0)
                deadline = time.monotonic() + 30
                while await _metric(http, url, "ssd_promotions_total") <= promotions_before:
                    assert time.monotonic() < deadline, "Shared SSD prefix was not promoted"
                    await generate([105], 0)
                    await asyncio.sleep(0.1)
