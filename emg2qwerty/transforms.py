# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
import torch
import torchaudio
import torch.nn.functional as F




TTransformIn = TypeVar("TTransformIn")
TTransformOut = TypeVar("TTransformOut")
Transform = Callable[[TTransformIn], TTransformOut]

def debug_shape(name, tensor):
    print(f"{name} shape: {tensor.shape}")


@dataclass
class ToTensor:
    """Extracts the specified ``fields`` from a numpy structured array
    and stacks them into a ``torch.Tensor``.

    Following TNC convention as a default, the returned tensor is of shape
    (time, field/batch, electrode_channel).

    Args:
        fields (list): List of field names to be extracted from the passed in
            structured numpy ndarray.
        stack_dim (int): The new dimension to insert while stacking
            ``fields``. (default: 1)
    """

    fields: Sequence[str] = ("emg_left", "emg_right")
    stack_dim: int = 1

    def __call__(self, data: np.ndarray) -> torch.Tensor:
        return torch.stack(
            [torch.as_tensor(data[f]) for f in self.fields], dim=self.stack_dim
        )


@dataclass
class Lambda:
    """Applies a custom lambda function as a transform.

    Args:
        lambd (lambda): Lambda to wrap within.
    """

    lambd: Transform[Any, Any]

    def __call__(self, data: Any) -> Any:
        return self.lambd(data)


@dataclass
class ForEach:
    """Applies the provided ``transform`` over each item of a batch
    independently. By default, assumes the input is of shape (T, N, ...).

    Args:
        transform (Callable): The transform to apply to each batch item of
            the input tensor.
        batch_dim (int): The bach dimension, i.e., the dim along which to
            unstack/unbind the input tensor prior to mapping over
            ``transform`` and restacking. (default: 1)
    """

    transform: Transform[torch.Tensor, torch.Tensor]
    batch_dim: int = 1

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [self.transform(t) for t in tensor.unbind(self.batch_dim)],
            dim=self.batch_dim,
        )


@dataclass
class Compose:
    """Compose a chain of transforms.

    Args:
        transforms (list): List of transforms to compose.
    """

    transforms: Sequence[Transform[Any, Any]]

    def __call__(self, data: Any) -> Any:
        for transform in self.transforms:
            data = transform(data)
        return data


@dataclass
class RandomBandRotation:
    """Applies band rotation augmentation by shifting the electrode channels
    by an offset value randomly chosen from ``offsets``. By default, assumes
    the input is of shape (..., C).

    NOTE: If the input is 3D with batch dim (TNC), then this transform
    applies band rotation for all items in the batch with the same offset.
    To apply different rotations each batch item, use the ``ForEach`` wrapper.

    Args:
        offsets (list): List of integers denoting the offsets by which the
            electrodes are allowed to be shift. A random offset from this
            list is chosen for each application of the transform.
        channel_dim (int): The electrode channel dimension. (default: -1)
    """

    offsets: Sequence[int] = (-1, 0, 1)
    channel_dim: int = -1

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        offset = np.random.choice(self.offsets) if len(self.offsets) > 0 else 0
        return tensor.roll(offset, dims=self.channel_dim)


@dataclass
class TemporalAlignmentJitter:
    """Applies a temporal jittering augmentation that randomly jitters the
    alignment of left and right EMG data by up to ``max_offset`` timesteps.
    The input must be of shape (T, ...).

    Args:
        max_offset (int): The maximum amount of alignment jittering in terms
            of number of timesteps.
        stack_dim (int): The dimension along which the left and right data
            are stacked. See ``ToTensor()``. (default: 1)
    """

    max_offset: int
    stack_dim: int = 1


    def __post_init__(self) -> None:
        assert self.max_offset >= 0

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        debug_shape("After TemporalJitter", tensor)
        assert tensor.shape[self.stack_dim] == 2
        left, right = tensor.unbind(self.stack_dim)

        offset = np.random.randint(-self.max_offset, self.max_offset + 1)
        if offset > 0:
            left = left[offset:]
            right = right[:-offset]
        if offset < 0:
            left = left[:offset]
            right = right[-offset:]

        return torch.stack([left, right], dim=self.stack_dim)


@dataclass
class LogSpectrogram:
    """Creates log10-scaled spectrogram from an EMG signal. In the case of
    multi-channeled signal, the channels are treated independently.
    The input must be of shape (T, ...) and the returned spectrogram
    is of shape (T, ..., freq).

    Args:
        n_fft (int): Size of FFT, creates n_fft // 2 + 1 frequency bins.
            (default: 64)
        hop_length (int): Number of samples to stride between consecutive
            STFT windows. (default: 16)
    """

    n_fft: int = 64
    hop_length: int = 16

    def __post_init__(self) -> None:
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            normalized=True,
            # Disable centering of FFT windows to avoid padding inconsistencies
            # between train and test (due to differing window lengths), as well
            # as to be more faithful to real-time/streaming execution.
            center=False,
        )

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        x = tensor.movedim(0, -1)  # (T, ..., C) -> (..., C, T)
        spec = self.spectrogram(x)  # (..., C, freq, T)
        logspec = torch.log10(spec + 1e-6)  # (..., C, freq, T)
        return logspec.movedim(-1, 0)  # (T, ..., C, freq)


@dataclass
class SpecAugment:
    """Applies time and frequency masking as per the paper
    "SpecAugment: A Simple Data Augmentation Method for Automatic Speech
    Recognition, Park et al" (https://arxiv.org/abs/1904.08779).

    Args:
        n_time_masks (int): Maximum number of time masks to apply,
            uniformly sampled from 0. (default: 0)
        time_mask_param (int): Maximum length of each time mask,
            uniformly sampled from 0. (default: 0)
        iid_time_masks (int): Whether to apply different time masks to
            each band/channel (default: True)
        n_freq_masks (int): Maximum number of frequency masks to apply,
            uniformly sampled from 0. (default: 0)
        freq_mask_param (int): Maximum length of each frequency mask,
            uniformly sampled from 0. (default: 0)
        iid_freq_masks (int): Whether to apply different frequency masks to
            each band/channel (default: True)
        mask_value (float): Value to assign to the masked columns (default: 0.)
    """

    n_time_masks: int = 0
    time_mask_param: int = 0
    iid_time_masks: bool = True
    n_freq_masks: int = 0
    freq_mask_param: int = 0
    iid_freq_masks: bool = True
    mask_value: float = 0.0

    def __post_init__(self) -> None:
        self.time_mask = torchaudio.transforms.TimeMasking(
            self.time_mask_param, iid_masks=self.iid_time_masks
        )
        self.freq_mask = torchaudio.transforms.FrequencyMasking(
            self.freq_mask_param, iid_masks=self.iid_freq_masks
        )

    def __call__(self, specgram: torch.Tensor) -> torch.Tensor:
        # (T, ..., C, freq) -> (..., C, freq, T)
        x = specgram.movedim(0, -1)

        # Time masks
        n_t_masks = np.random.randint(self.n_time_masks + 1)
        for _ in range(n_t_masks):
            x = self.time_mask(x, mask_value=self.mask_value)

        # Frequency masks
        n_f_masks = np.random.randint(self.n_freq_masks + 1)
        for _ in range(n_f_masks):
            x = self.freq_mask(x, mask_value=self.mask_value)

        # (..., C, freq, T) -> (T, ..., C, freq)
        return x.movedim(-1, 0)


@dataclass
class RandomCrop:
    min_crop_size: int
    max_crop_size: int
    n_fft: int = 64  # STFT requires T >= n_fft

    def __post_init__(self):
        assert self.min_crop_size >= self.n_fft, f"min_crop_size ({self.min_crop_size}) must be >= n_fft ({self.n_fft})"
        assert self.max_crop_size >= self.min_crop_size, "max_crop_size must be >= min_crop_size"

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Randomly crops the time dimension while ensuring T_crop >= n_fft."""
        if tensor.ndim < 3:
            raise ValueError(f"Expected at least 3D input (T, B, C) or (T, N, B, C), got {tensor.shape}")

        T = tensor.shape[0]
        min_valid_crop = max(self.min_crop_size, self.n_fft)

        if T < min_valid_crop:
            pad_amount = min_valid_crop - T
            tensor = torch.nn.functional.pad(tensor, (0, 0, 0, 0, 0, pad_amount))  # Pad time dimension

        crop_size = np.random.randint(min_valid_crop, min(self.max_crop_size, T) + 1)
        start_idx = np.random.randint(0, T - crop_size + 1)
        cropped_tensor = tensor[start_idx : start_idx + crop_size]

        return cropped_tensor



@dataclass
class EnsureValidTDim:
    """Ensures the time dimension (T) is at least n_fft before STFT."""

    n_fft: int = 64  # Match the STFT requirement

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        T = tensor.shape[0]
        if T < self.n_fft:
            pad_amount = self.n_fft - T
            tensor = F.pad(tensor, (0, 0, 0, 0, 0, pad_amount))  # Pad time dimension
        return tensor
    
@dataclass
class GaussianNoise:
    """Applies Gaussian noise to the input tensor.

    Args:
        mean (float): Mean of the Gaussian noise distribution. (default: 0.0)
        std (float): Standard deviation of the Gaussian noise distribution.
        apply_prob (float): Probability of applying noise to a sample. (default: 1.0)
    """

    mean: float = 0.0
    std: float = 0.01  # Adjust this based on noise tolerance for EMG
    apply_prob: float = 1.0

    def __post_init__(self) -> None:
        assert self.std >= 0, "Standard deviation must be non-negative."
        assert 0.0 <= self.apply_prob <= 1.0, "apply_prob must be between 0 and 1."

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if np.random.rand() > self.apply_prob:
            return tensor  # No noise added with probability (1 - apply_prob)

        noise = torch.normal(mean=self.mean, std=self.std, size=tensor.shape)
        return tensor + noise


@dataclass
class RandomMasking:
    """Applies random masking along the time or channel dimension.

    Args:
        mask_prob (float): Probability of applying masking to a sample. (default: 1.0)
        mask_ratio (float): Percentage of values to mask (0 to 1). (default: 0.1)
        mask_dim (str): Dimension to apply masking ('time' or 'channel'). (default: 'time')
        mask_value (float): Value to assign to the masked regions. (default: 0.0)
    """

    mask_prob: float = 1.0
    mask_ratio: float = 0.1
    mask_dim: str = "time"  # Can be "time" or "channel"
    mask_value: float = 0.0

    def __post_init__(self) -> None:
        assert 0.0 <= self.mask_prob <= 1.0, "mask_prob must be between 0 and 1."
        assert 0.0 <= self.mask_ratio <= 1.0, "mask_ratio must be between 0 and 1."
        assert self.mask_dim in ["time", "channel"], "mask_dim must be 'time' or 'channel'."

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if np.random.rand() > self.mask_prob:
            return tensor  # No masking applied

        masked_tensor = tensor.clone()

        if self.mask_dim == "time":
            # Mask a percentage of time steps
            num_time_steps = tensor.shape[0]
            num_mask = int(self.mask_ratio * num_time_steps)
            mask_indices = np.random.choice(num_time_steps, num_mask, replace=False)
            masked_tensor[mask_indices, ...] = self.mask_value

        elif self.mask_dim == "channel":
            # Mask a percentage of channels
            num_channels = tensor.shape[-1]
            num_mask = int(self.mask_ratio * num_channels)
            mask_indices = np.random.choice(num_channels, num_mask, replace=False)
            masked_tensor[..., mask_indices] = self.mask_value

        return masked_tensor
