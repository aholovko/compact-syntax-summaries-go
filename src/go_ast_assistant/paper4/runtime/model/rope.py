from __future__ import annotations

from typing import Any

import torch


def precompute_rope_params(
    head_dim: int,
    theta_base: float = 500_000.0,
    context_length: int = 8192,
    freq_config: dict[str, Any] | None = None,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert head_dim % 2 == 0, "head_dim must be even"
    table_dtype = torch.get_default_dtype() if dtype is None else dtype
    table_device = torch.device("cpu") if device is None else device

    indices = torch.arange(head_dim // 2, device=table_device, dtype=table_dtype)
    inverse_frequencies = theta_base ** (-2.0 * indices / head_dim)
    if freq_config is not None:
        original_length = freq_config["original_context_length"]
        low_wavelength = original_length / freq_config["low_freq_factor"]
        high_wavelength = original_length / freq_config["high_freq_factor"]
        wavelength = 2 * torch.pi / inverse_frequencies
        scaled = torch.where(
            wavelength > low_wavelength,
            inverse_frequencies / freq_config["factor"],
            inverse_frequencies,
        )
        smooth = (original_length / wavelength - freq_config["low_freq_factor"]) / (
            freq_config["high_freq_factor"] - freq_config["low_freq_factor"]
        )
        smoothed = (1 - smooth) * (inverse_frequencies / freq_config["factor"]) + smooth * inverse_frequencies
        medium = (wavelength <= low_wavelength) & (wavelength >= high_wavelength)
        inverse_frequencies = torch.where(medium, smoothed, scaled)

    positions = torch.arange(context_length, device=table_device, dtype=table_dtype)
    half_angles = positions.unsqueeze(1) * inverse_frequencies.unsqueeze(0)
    angles = torch.cat((half_angles, half_angles), dim=1)
    return torch.cos(angles), torch.sin(angles)


def compute_rope(
    values: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    offset: int = 0,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    assert values.dim() == 4, "values must be (batch, heads, seq_len, head_dim)"
    sequence_length = values.shape[2]
    head_dim = values.shape[3]
    assert head_dim % 2 == 0, "head_dim must be even"

    if position_ids is None:
        selected_cos = cos[offset : offset + sequence_length].unsqueeze(0).unsqueeze(0)
        selected_sin = sin[offset : offset + sequence_length].unsqueeze(0).unsqueeze(0)
    else:
        selected_cos = cos[position_ids].unsqueeze(1)
        selected_sin = sin[position_ids].unsqueeze(1)
    selected_cos = selected_cos.to(dtype=values.dtype, device=values.device)
    selected_sin = selected_sin.to(dtype=values.dtype, device=values.device)

    first_half, second_half = values[..., : head_dim // 2], values[..., head_dim // 2 :]
    rotated = torch.cat((-second_half, first_half), dim=-1)
    return (values * selected_cos + rotated * selected_sin).to(values.dtype)
