"""BN+SiLU fused CUDA op (optimized: vectorized + warp shuffle + inplace SiLU)."""
import torch
from torch.utils.cpp_extension import load

_ext = load(
    name="bn_silu_fused",
    sources=["/home/lee/Documents/GitHub/YOLOX/yolox/layers/bn_silu/bn_silu_fused.cu"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
    verbose=False,
)


class BNSiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, running_mean, running_var, eps, training, momentum):
        output, save_mean, save_var = _ext.forward(
            input, weight, bias, running_mean, running_var, eps, training, momentum
        )
        if training:
            ctx.save_for_backward(input, save_mean, save_var, weight, bias)
            ctx.eps = eps
            ctx.shape = input.shape
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, save_mean, save_var, weight, bias = ctx.saved_tensors
        N, C, H, W = ctx.shape
        ginput, gw, gb = _ext.backward(
            grad_output, input, save_mean, save_var,
            weight, bias, ctx.eps, N, C, H, W,
        )
        return ginput, gw, gb, None, None, None, None, None


def bn_silu(input, weight, bias, running_mean, running_var, eps=1e-5, training=True, momentum=0.1):
    return BNSiLU.apply(input, weight, bias, running_mean, running_var, eps, training, momentum)
