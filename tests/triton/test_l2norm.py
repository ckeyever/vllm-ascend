import torch
from vllm_ascend.ops.triton.fla.l2norm import l2norm_fwd as l2norm
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

def func():
    torch.manual_seed(42)
    init_device_properties_triton()
    device = "npu"
    x = torch.randn(4, 128, 128, 100, dtype=torch.float16).to(device).requires_grad_(False)
    npu_out = l2norm(x)
    print(npu_out)

func()