"""Parameter-free separable three-axis rotary embeddings."""

from __future__ import annotations

import math

import torch


def grid_coordinates(time: int, height: int, width: int, *, device: torch.device) -> torch.Tensor:
    """Return normalized ``[T,H,W,3]`` coordinates; a singleton axis is exactly zero."""

    def axis(size: int) -> torch.Tensor:
        if size <= 0:
            raise ValueError("grid axes must be positive")
        if size == 1:
            return torch.zeros(1, device=device, dtype=torch.float32)
        return torch.linspace(-1.0, 1.0, size, device=device, dtype=torch.float32)

    t, h, w = torch.meshgrid(axis(time), axis(height), axis(width), indexing="ij")
    return torch.stack((t, h, w), dim=-1)


def _rotate_axis(value: torch.Tensor, coordinate: torch.Tensor, axis_dim: int) -> torch.Tensor:
    pairs = axis_dim // 2
    heads = value.shape[-2]
    frequencies = torch.logspace(
        math.log10(math.pi),
        math.log10(10.0 * math.pi),
        heads * pairs,
        device=value.device,
        dtype=torch.float32,
    ).reshape(pairs, heads).transpose(0, 1)
    phase = coordinate.to(torch.float32).unsqueeze(-1) * frequencies
    cos = phase.cos().to(value.dtype)
    sin = phase.sin().to(value.dtype)
    shaped = value.reshape(*value.shape[:-1], pairs, 2)
    first, second = shaped.unbind(-1)
    return torch.stack((first * cos - second * sin, second * cos + first * sin), dim=-1).flatten(-2)


def apply_separable_rope(
    value: torch.Tensor,
    coordinates: torch.Tensor,
    axis_dims: tuple[int, int, int],
) -> torch.Tensor:
    """Rotate heads-last Q/K tensors over time, height and width independently."""

    if value.shape[-1] != sum(axis_dims):
        raise ValueError(f"head dim {value.shape[-1]} does not match RoPE allocation {axis_dims}")
    if coordinates.shape[-1] != 3 or tuple(value.shape[1:-2]) != tuple(coordinates.shape[:-1]):
        raise ValueError(
            f"coordinates {tuple(coordinates.shape)} do not match heads-last grid {tuple(value.shape)}"
        )
    coordinate = coordinates.to(device=value.device)
    while coordinate.ndim < value.ndim - 1:
        coordinate = coordinate.unsqueeze(0)
    chunks = value.split(axis_dims, dim=-1)
    return torch.cat(
        [
            _rotate_axis(chunk, coordinate[..., axis].unsqueeze(-1), axis_dim)
            for axis, (chunk, axis_dim) in enumerate(zip(chunks, axis_dims))
        ],
        dim=-1,
    )
