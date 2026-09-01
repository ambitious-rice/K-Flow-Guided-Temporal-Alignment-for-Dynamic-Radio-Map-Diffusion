"""Joint global attention and true sliding 2-D/3-D neighborhood attention."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product

import torch
import torch.nn.functional as F
from torch import nn

from .common import xavier_linear
from .rope import apply_separable_rope


def _scaled_cosine_qk(q: torch.Tensor, k: torch.Tensor, logit_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = logit_scale.float().exp().clamp(max=100.0).sqrt().to(q.dtype)
    shape = *((1,) * (q.ndim - 2)), scale.shape[0], 1
    factor = scale.reshape(shape)
    return F.normalize(q.float(), dim=-1).to(q.dtype) * factor, F.normalize(k.float(), dim=-1).to(k.dtype) * factor


def reference_neighborhood_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: Sequence[int],
) -> torch.Tensor:
    """Small-tensor correctness backend; production is required to use NATTEN."""

    dimensions = len(q.shape[1:-2])
    kernel = tuple(int(value) for value in kernel_size)
    grid_shape = tuple(int(value) for value in q.shape[1:-2])
    if len(kernel) != dimensions or any(value <= 0 or value % 2 == 0 for value in kernel):
        raise ValueError("reference neighborhood kernels must be positive odd values matching the grid rank")
    if any(kernel_axis > grid_axis for kernel_axis, grid_axis in zip(kernel, grid_shape)):
        raise ValueError("reference neighborhood kernel cannot exceed its token-grid axis")
    neighbors_by_axis = []
    for grid_axis, kernel_axis in zip(grid_shape, kernel):
        query = torch.arange(grid_axis, device=q.device)
        start = (query - kernel_axis // 2).clamp(0, grid_axis - kernel_axis)
        neighbors_by_axis.append(start[:, None] + torch.arange(kernel_axis, device=q.device)[None, :])
    shifted_keys, shifted_values = [], []
    for neighbor_position in product(*(range(size) for size in kernel)):
        axes = [neighbors[:, position] for neighbors, position in zip(neighbors_by_axis, neighbor_position)]
        indices = torch.meshgrid(*axes, indexing="ij")
        selection = (slice(None), *indices, slice(None), slice(None))
        shifted_keys.append(k[selection])
        shifted_values.append(v[selection])
    keys = torch.stack(shifted_keys, dim=-2)
    values = torch.stack(shifted_values, dim=-2)
    scores = (q.unsqueeze(-2) * keys).sum(dim=-1, keepdim=True)
    weights = scores.float().softmax(dim=-2).to(v.dtype)
    return (weights * values).sum(dim=-2)


def _natten_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kernel: tuple[int, ...]) -> torch.Tensor:
    try:
        import natten
    except ImportError as error:
        raise ModuleNotFoundError("HV-DiT V4 requires pinned NATTEN for production local attention") from error
    operation_name = "na2d" if len(kernel) == 2 else "na3d"
    operation = getattr(natten, operation_name, None)
    if operation is None:
        raise RuntimeError(f"Installed NATTEN does not expose required {operation_name}")
    if not bool(getattr(natten, "HAS_LIBNATTEN", False)):
        raise RuntimeError("Pinned NATTEN is present but libnatten CUDA kernels are unavailable")
    return operation(
        q,
        k,
        v,
        kernel_size=kernel,
        stride=1,
        is_causal=False,
        scale=1.0,
        backend="cutlass-fna",
    )


class NeighborhoodSelfAttention(nn.Module):
    """Attention transform only; the enclosing DiT block owns the residual gate."""

    def __init__(
        self,
        dim: int,
        head_dim: int,
        *,
        kernel_size: tuple[int, ...],
        rope_axis_dims: tuple[int, int, int],
        backend: str,
        dropout: float,
    ) -> None:
        super().__init__()
        if dim % head_dim:
            raise ValueError("attention dim must be divisible by head_dim")
        self.heads = dim // head_dim
        self.head_dim = int(head_dim)
        self.kernel_size = tuple(int(value) for value in kernel_size)
        self.rope_axis_dims = tuple(int(value) for value in rope_axis_dims)
        self.backend = str(backend)
        self.qkv = xavier_linear(nn.Linear(dim, 3 * dim, bias=False))
        self.logit_scale = nn.Parameter(torch.full((self.heads,), torch.log(torch.tensor(10.0))))
        self.dropout = nn.Dropout(dropout)
        self.projection = xavier_linear(nn.Linear(dim, dim, bias=False))

    def forward(self, state: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        restore_singleton_time = False
        if len(self.kernel_size) == 2:
            if state.ndim != 5 or state.shape[1] != 1 or coordinates.shape[0] != 1:
                raise ValueError("2-D neighborhood attention requires an explicit singleton time axis")
            state = state[:, 0]
            coordinates = coordinates[0]
            restore_singleton_time = True
        qkv = self.qkv(state).reshape(*state.shape[:-1], 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(-3)
        q, k = _scaled_cosine_qk(q, k, self.logit_scale)
        q = apply_separable_rope(q, coordinates, self.rope_axis_dims)
        k = apply_separable_rope(k, coordinates, self.rope_axis_dims)
        if self.backend == "natten":
            attended = _natten_attention(q, k, v, self.kernel_size)
        elif self.backend == "reference":
            attended = reference_neighborhood_attention(q, k, v, self.kernel_size)
        else:
            raise ValueError(f"Unsupported neighborhood attention backend: {self.backend}")
        attended = self.projection(self.dropout(attended.reshape(*state.shape[:-1], -1)))
        return attended.unsqueeze(1) if restore_singleton_time else attended


class GlobalSpaceTimeSelfAttention(nn.Module):
    """Joint attention over the flattened ``T*H*W`` bottleneck tokens."""

    def __init__(
        self,
        dim: int,
        head_dim: int,
        *,
        rope_axis_dims: tuple[int, int, int],
        dropout: float,
    ) -> None:
        super().__init__()
        if dim % head_dim:
            raise ValueError("attention dim must be divisible by head_dim")
        self.heads = dim // head_dim
        self.head_dim = int(head_dim)
        self.rope_axis_dims = tuple(int(value) for value in rope_axis_dims)
        self.qkv = xavier_linear(nn.Linear(dim, 3 * dim, bias=False))
        self.logit_scale = nn.Parameter(torch.full((self.heads,), torch.log(torch.tensor(10.0))))
        self.dropout = nn.Dropout(dropout)
        self.projection = xavier_linear(nn.Linear(dim, dim, bias=False))

    def forward(self, state: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        batch, time, height, width, dim = state.shape
        qkv = self.qkv(state).reshape(batch, time, height, width, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(-3)
        q, k = _scaled_cosine_qk(q, k, self.logit_scale)
        q = apply_separable_rope(q, coordinates, self.rope_axis_dims)
        k = apply_separable_rope(k, coordinates, self.rope_axis_dims)
        tokens = time * height * width
        q = q.reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout.p if self.training else 0.0,
            scale=1.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, time, height, width, dim)
        return self.projection(attended)
