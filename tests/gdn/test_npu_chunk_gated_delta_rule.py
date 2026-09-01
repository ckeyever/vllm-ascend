"""Standalone invocation for ``torch.ops._C_ascend.npu_chunk_gated_delta_rule``.

Run this file on an Ascend NPU from the ``vllm-ascend`` project directory::

    python tests/gdn/test_npu_chunk_gated_delta_rule.py

The tensors use the TND layout expected by the operator.  The two entries in
``actual_seq_lengths`` describe the two sequences concatenated along the
token dimension.
"""

from __future__ import annotations

import argparse

import torch
import torch_npu  # noqa: F401  # Registers the NPU device backend.

from vllm_ascend.utils import enable_custom_op


TOTAL_TOKENS = 8088
NUM_QK_HEADS = 4
NUM_V_HEADS = 8
KEY_DIM = 128
VALUE_DIM = 128
NUM_SEQUENCES = 2


def run_npu_chunk_gated_delta_rule(
    device: torch.device | str = "npu:0",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the requested inputs and invoke the Ascend operator once."""

    if not enable_custom_op():
        raise RuntimeError(
            "vllm-ascend custom operators are disabled in this environment. "
            "Unset VLLM_BATCH_INVARIANT and use a supported Ascend device."
        )

    torch.manual_seed(0)
    npu = torch.device(device)

    # query/key/value/beta/initial_state are BF16; g is FLOAT (float32).
    query = torch.randn(
        TOTAL_TOKENS, NUM_QK_HEADS, KEY_DIM, dtype=torch.bfloat16, device=npu
    ).contiguous()
    key = torch.randn(
        TOTAL_TOKENS, NUM_QK_HEADS, KEY_DIM, dtype=torch.bfloat16, device=npu
    ).contiguous()
    value = torch.randn(
        TOTAL_TOKENS, NUM_V_HEADS, VALUE_DIM, dtype=torch.bfloat16, device=npu
    ).contiguous()
    beta = torch.rand(
        TOTAL_TOKENS, NUM_V_HEADS, dtype=torch.bfloat16, device=npu
    ).contiguous()
    initial_state = torch.randn(
        NUM_SEQUENCES,
        NUM_V_HEADS,
        VALUE_DIM,
        KEY_DIM,
        dtype=torch.bfloat16,
        device=npu,
    ).contiguous()
    actual_seq_lengths = torch.tensor(
        [TOTAL_TOKENS // NUM_SEQUENCES] * NUM_SEQUENCES,
        dtype=torch.int32,
        device=npu,
    ).contiguous()
    g_optional = torch.randn(
        TOTAL_TOKENS, NUM_V_HEADS, dtype=torch.float32, device=npu
    ).contiguous()

    # The scale used by the GDN attention path is query head_dim ** -0.5.
    scale_value = KEY_DIM**-0.5
    out, final_state = torch.ops._C_ascend.npu_chunk_gated_delta_rule(
        query,
        key,
        value,
        beta,
        initial_state,
        actual_seq_lengths,
        g_optional,
        scale_value,
    )
    return out, final_state


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Invoke npu_chunk_gated_delta_rule with the requested shapes."
    )
    parser.add_argument(
        "--device",
        default="npu:0",
        help="NPU device used for inputs and execution (default: npu:0).",
    )
    args = parser.parse_args()

    out, final_state = run_npu_chunk_gated_delta_rule(args.device)
    expected_out_shape = (TOTAL_TOKENS, NUM_V_HEADS, VALUE_DIM)
    expected_state_shape = (NUM_SEQUENCES, NUM_V_HEADS, VALUE_DIM, KEY_DIM)
    if tuple(out.shape) != expected_out_shape:
        raise RuntimeError(f"Unexpected out shape: {tuple(out.shape)}")
    if tuple(final_state.shape) != expected_state_shape:
        raise RuntimeError(f"Unexpected final_state shape: {tuple(final_state.shape)}")

    print(f"out: {tuple(out.shape)}, dtype={out.dtype}, device={out.device}")
    print(
        "final_state: "
        f"{tuple(final_state.shape)}, dtype={final_state.dtype}, device={final_state.device}"
    )


if __name__ == "__main__":
    main()
