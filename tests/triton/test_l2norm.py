import torch
import torch_npu
from vllm_ascend.ops.triton.fla.l2norm import l2norm_fwd as l2norm
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

def func(T, H):
    torch.manual_seed(42)
    init_device_properties_triton()
    device = "npu"
    x = torch.randn(T, H, dtype=torch.float16).to(device).requires_grad_(False)
    npu_out = l2norm(x)
    torch.npu.synchronize()

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        msprof_tx=False,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        l2_cache=False,
        op_attr=False,
        data_simplification=True,
        record_op_args=False,
    )

    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        with_stack=False,
        profile_memory=False,
        experimental_config=experimental_config,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            "./l2norm_profile",
            worker_name="l2norm",
        ),
    ) as prof:
        for _ in range(20):
            l2norm(x)
            prof.step()
    torch.npu.synchronize()

func(16, 128)
func(4096, 128)

func(16, 110)
func(4096, 110)