import torch
import torch.types
import typing as t

def length_mask(max_length: int, length: torch.Tensor) -> torch.Tensor:
    """
    Args:
        max_length: The maximum length of the mask.
        length: The length of each element in the batch. Shape: (N,)

    The mask is True for the elements that are within the length.

    The mask is of shape (N, max_length).
    """
    return torch.arange(max_length, device=length.device) < length.unsqueeze(-1)


def pad(
    x, max_len: int, pad_value: torch.types.Number = 0, dim: int = 0
) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    if x.shape[dim] == max_len:
        return x
    if x.shape[dim] > max_len:
        return x.narrow(dim, 0, max_len)
    pad_shape = list(x.shape)
    pad_shape[dim] = max_len - x.shape[dim]
    padding = torch.full(pad_shape, pad_value, device=x.device, dtype=x.dtype)
    return torch.concat([x, padding], dim)


def pad_batch(
    x: list[torch.Tensor],
    max_len: t.Union[int, None] = None,
    pad_value: torch.types.Number = 0,
    dim: int = 0,
) -> torch.Tensor:
    if max_len is None:
        max_len = max(y.shape[dim] for y in x)
    return torch.stack([pad(y, max_len, pad_value, dim) for y in x])
