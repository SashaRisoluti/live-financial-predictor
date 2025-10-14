"""Utilities per conversioni tensor comuni."""

import numpy as np
import torch
from typing import Union


def to_tensor(
    data: Union[np.ndarray, list, torch.Tensor],
    device: str = "cpu",
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Converti a tensor PyTorch."""
    
    if isinstance(data, torch.Tensor):
        return data.to(device=device, dtype=dtype)
    
    return torch.tensor(data, device=device, dtype=dtype)


def to_numpy(data: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Converti a numpy array."""
    
    if isinstance(data, torch.Tensor):
        return data.cpu().detach().numpy()
    
    return np.array(data)


def ensure_batch_dim(
    tensor: torch.Tensor,
    batch_first: bool = True
) -> torch.Tensor:
    """Assicura che tensor abbia dimensione batch."""
    
    if tensor.ndim == 1:
        # [length] → [1, length]
        return tensor.unsqueeze(0)
    elif tensor.ndim == 2 and not batch_first:
        # [length, features] → [1, length, features]
        return tensor.unsqueeze(0)
    
    return tensor


def pad_or_truncate(
    tensor: torch.Tensor,
    target_length: int,
    dim: int = -1,
    pad_value: float = 0.0
) -> torch.Tensor:
    """Padda o tronca tensor a lunghezza target."""
    
    current_length = tensor.shape[dim]
    
    if current_length == target_length:
        return tensor
    elif current_length > target_length:
        # Truncate (prendi ultimi valori)
        if dim == -1 or dim == tensor.ndim - 1:
            return tensor[..., -target_length:]
        else:
            indices = [slice(None)] * tensor.ndim
            indices[dim] = slice(-target_length, None)
            return tensor[tuple(indices)]
    else:
        # Pad
        pad_length = target_length - current_length
        pad_shape = list(tensor.shape)
        pad_shape[dim] = pad_length
        
        padding = torch.full(
            pad_shape,
            pad_value,
            dtype=tensor.dtype,
            device=tensor.device
        )
        
        return torch.cat([padding, tensor], dim=dim)
