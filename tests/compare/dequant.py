"""INT4 dequantization for AutoRound / AutoGPTQ format."""

import numpy as np
import torch


def unpack_int4(data: torch.Tensor) -> torch.Tensor:
    """Unpack INT4 values from int32-packed tensor.

    Input: [M, N/8] int32 where each int32 packs 8 int4 values.
    Output: [M, N] float with each element in [0, 15].
    """
    M, N_packed = data.shape
    raw_bytes = data.numpy().astype(np.int32).view(np.uint8).reshape(M, N_packed * 4)
    low = raw_bytes & 0x0F
    high = (raw_bytes >> 4) & 0x0F
    result = np.zeros((M, N_packed * 8), dtype=np.uint8)
    result[:, 0::2] = low
    result[:, 1::2] = high
    return torch.from_numpy(result.astype(np.float32))


def dequantize_int4_autogptq(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Dequantize AutoRound/AutoGPTQ INT4 weights.

    The safetensors store transposed weight layout:
        qweight: [K/8, N] for gate_proj/up_proj (transposed from [N, K])
        qzeros:  [num_groups, N/8]
        scales:  [num_groups, N]

    The weight data layout is:
        qweight[k_packed][n] = uint32 packing 8 INT4 values for
        K positions k_packed*8..k_packed*8+7 at output feature n.

    After unpack_int4, the shape is [K_packed, N*8]. We must reshape
    to [K_packed, N, 8] then permute to [K_packed, 8, N] to get the
    correct [K, N] layout where element [k][n] = weight[k][n].

    The dequant formula is: w_deq = (w_int4 - zero_point) * scale
    where zero_point is the stored value in qzeros.

    For gate_proj with hidden_size=5120, intermediate_size=17408:
        qweight shape: [640, 17408] = [K/8, N]
        After correct unpack + permute: [5120, 17408] = [K, N]
    """
    K_packed_dim = qweight.shape[0]
    N_dim = qweight.shape[1]
    K_dim = K_packed_dim * 8

    # Unpack weights: [K/8, N] int32 -> [K_packed, N*8] float
    w_int4 = unpack_int4(qweight)
    z_int4 = unpack_int4(qzeros)

    num_groups = scales.shape[0]
    if K_dim % group_size != 0:
        raise ValueError(f"K={K_dim} not divisible by group_size={group_size}")

    # CRITICAL: Fix the layout from [K_packed, N*8] to [K, N].
    # Each uint32 packed 8 values along K, interleaved with N.
    # Correct: reshape -> permute -> flatten
    #   [K_packed, N, 8] -> [K_packed, 8, N] -> [K, N]
    w_correct = w_int4.reshape(K_packed_dim, N_dim, 8).permute(0, 2, 1).reshape(K_dim, N_dim)
    z_correct = z_int4.reshape(-1, N_dim, 8).permute(0, 2, 1).reshape(-1, N_dim)

    # Reshape for per-group dequant along K axis
    w_grps = w_correct.reshape(num_groups, group_size, N_dim)
    z_grps = z_correct.reshape(num_groups, 1, N_dim)
    s_grps = scales.reshape(num_groups, 1, N_dim)

    # Dequant: (w_int4 - zero_point) * scale
    w_f32 = (w_grps.float() - z_grps.float()) * s_grps.float()
    return w_f32.reshape(K_dim, N_dim)
