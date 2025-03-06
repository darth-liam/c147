# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence

import torch
from torch import nn


class SpectrogramNorm(nn.Module):
    """A `torch.nn.Module` that applies 2D batch normalization over spectrogram
    per electrode channel per band. Inputs must be of shape
    (T, N, num_bands, electrode_channels, frequency_bins).

    With left and right bands and 16 electrode channels per band, spectrograms
    corresponding to each of the 2 * 16 = 32 channels are normalized
    independently using `nn.BatchNorm2d` such that stats are computed
    over (N, freq, time) slices.

    Args:
        channels (int): Total number of electrode channels across bands
            such that the normalization statistics are calculated per channel.
            Should be equal to num_bands * electrode_chanels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape  # (T, N, bands=2, C=16, freq)
        assert self.channels == bands * C

        x = inputs.movedim(0, -1)  # (N, bands=2, C=16, freq, T)
        x = x.reshape(N, bands * C, freq, T)
        x = self.batch_norm(x)
        x = x.reshape(N, bands, C, freq, T)
        return x.movedim(-1, 0)  # (T, N, bands=2, C=16, freq)


class RotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that takes an input tensor of shape
    (T, N, electrode_channels, ...) corresponding to a single band, applies
    an MLP after shifting/rotating the electrodes for each positional offset
    in ``offsets``, and pools over all the outputs.

    Returns a tensor of shape (T, N, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input of
            shape (T, N, C, ...), this should be equal to C * ... (that is,
            the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        mlp: list[nn.Module] = []
        for out_features in mlp_features:
            mlp.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                ]
            )
            in_features = out_features
        self.mlp = nn.Sequential(*mlp)

        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"
        self.pooling = pooling

        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, C, ...)

        # Create a new dim for band rotation augmentation with each entry
        # corresponding to the original tensor with its electrode channels
        # shifted by one of ``offsets``:
        # (T, N, C, ...) -> (T, N, rotation, C, ...)
        x = torch.stack([x.roll(offset, dims=2) for offset in self.offsets], dim=2)

        # Flatten features and pass through MLP:
        # (T, N, rotation, C, ...) -> (T, N, rotation, mlp_features[-1])
        x = self.mlp(x.flatten(start_dim=3))

        # Pool over rotations:
        # (T, N, rotation, mlp_features[-1]) -> (T, N, mlp_features[-1])
        if self.pooling == "max":
            return x.max(dim=2).values
        else:
            return x.mean(dim=2)


class MultiBandRotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that applies a separate instance of
    `RotationInvariantMLP` per band for inputs of shape
    (T, N, num_bands, electrode_channels, ...).

    Returns a tensor of shape (T, N, num_bands, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input
            of shape (T, N, num_bands, C, ...), this should be equal to
            C * ... (that is, the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
        num_bands (int): ``num_bands`` for an input of shape
            (T, N, num_bands, C, ...). (default: 2)
        stack_dim (int): The dimension along which the left and right data
            are stacked. (default: 2)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim

        # One MLP per band
        self.mlps = nn.ModuleList(
            [
                RotationInvariantMLP(
                    in_features=in_features,
                    mlp_features=mlp_features,
                    pooling=pooling,
                    offsets=offsets,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands

        inputs_per_band = inputs.unbind(self.stack_dim)
        outputs_per_band = [
            mlp(_input) for mlp, _input in zip(self.mlps, inputs_per_band)
        ]
        return torch.stack(outputs_per_band, dim=self.stack_dim)


class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.width = width

        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        # TNC -> NCT -> NcwT
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

        # Skip connection after downsampling
        T_out = x.shape[0]
        x = x + inputs[-T_out:]

        # Layer norm over C
        return self.layer_norm(x)  # TNC


class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # TNC
        x = self.fc_block(x)
        x = x + inputs
        return self.layer_norm(x)  # TNC


class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0
        tds_conv_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, num_features // channels, kernel_width),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)
    
    
class SimpleCNN2dBlock(nn.Module):
    """A basic convolutional neural network for processing EMG spectrograms."""
    
    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.width = width
        
        self.conv2d = nn.Conv2d(
            in_channels=channels, 
            out_channels=channels, 
            kernel_size=(1,kernel_width),
            #output (T,N,num_features-kernal_width+1)
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels*width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in,N,C = inputs.shape

        x = inputs.movedim(0,-1).reshape(N,self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x) 
        x = x.reshape(N, C, -1).movedim(-1,0)

        T_out = x.shape[0]
        x = x + inputs[-T_out:]

        return self.layer_norm(x)
    

class MultiLayerCNNBlock(nn.Module):
    """A 4-layer CNN block for processing EMG spectrograms."""
    
    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.width = width
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, kernel_width), padding=(0, kernel_width // 2 - 1)),
            nn.ReLU(),
            nn.BatchNorm2d(channels),

            nn.Conv2d(channels, channels, kernel_size=(1, kernel_width), padding=(0, kernel_width // 2 - 1)),
            nn.ReLU(),
            nn.BatchNorm2d(channels),

            nn.Conv2d(channels, channels, kernel_size=(1, kernel_width), padding=(0, kernel_width // 2 - 1)),
            nn.ReLU(),
            nn.BatchNorm2d(channels),
        )

        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape

        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv_layers(x)  # Pass through 4 convolutional layers
        x = x.reshape(N, C, -1).movedim(-1, 0)
        T_out = x.shape[0]
        x = x + inputs[-T_out:]

        return self.layer_norm(x)
    
class LSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, bidirectional=True, dropout=0.3):
        super(LSTMBlock, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=bidirectional,
            dropout=dropout
        )
        self.layer_norm = nn.LayerNorm(input_size)
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: (T, N, C) - (Time, Batch, Channels)
        Outputs: Same shape as input (T, N, C)
        """
        T_in, N, C = inputs.shape
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(inputs)  # Shape: (T, N, hidden_size * num_directions)
        
        # Residual connection (aligning dimensions if needed)
        if lstm_out.shape[-1] != inputs.shape[-1]:
            # Make sure dimensions match by using a linear layer to transform the LSTM output
            lstm_out = self.fc(lstm_out)
        
        # Apply residual connection (alignment is taken care of here)
        x = lstm_out + inputs
        
        # Normalize and return
        return self.layer_norm(x)

    

class CNNFCBlock(nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        x = self.fc_block(x)
        x = x + inputs
        return self.layer_norm(x)
    

class LSTMFCBlock(nn.Module):
    """A fully connected block for LSTM feature refinement."""
    
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: (T, N, C) - (Time, Batch, Features)
        Outputs: Same shape as input (T, N, C)
        """
        x = inputs
        x = self.fc_block(x)
        x = x + inputs  # Residual connection
        return self.layer_norm(x)



class SimpleCNNEncoder(nn.Module):
        def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
            super().__init__()

            assert len(block_channels) > 0
            simple_cnn_conv_blocks: list[nn.Module] = []
            for channels in block_channels:
                assert (
                    num_features % channels == 0
                ), "block_channels must evenly divide num_features"
                simple_cnn_conv_blocks.extend(
                    [
                        SimpleCNN2dBlock(channels, num_features // channels, kernel_width),
                        CNNFCBlock(num_features),
                    ]
                )
            self.simple_cnn_conv_blocks = nn.Sequential(*simple_cnn_conv_blocks)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.simple_cnn_conv_blocks(inputs)  # (T, N, num_features)
    

class MultiLayerCNNEncoder(nn.Module):
        def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
            super().__init__()

            assert len(block_channels) > 0
            multi_layer_cnn_conv_blocks: list[nn.Module] = []
            for channels in block_channels:
                assert (
                    num_features % channels == 0
                ), "block_channels must evenly divide num_features"
                multi_layer_cnn_conv_blocks.extend(
                    [
                        MultiLayerCNNBlock(channels, num_features // channels, kernel_width),
                        CNNFCBlock(num_features),
                    ]
                )
            self.multi_layer_cnn_conv_blocks = nn.Sequential(*multi_layer_cnn_conv_blocks)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.multi_layer_cnn_conv_blocks(inputs)  # (T, N, num_features)
        

class LSTMEncoder(nn.Module):
    """An LSTM-based encoder for processing sequential EMG features."""

    def __init__(
        self,
        num_features: int,
        hidden_sizes: Sequence[int] = (24, 24, 24, 24),
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        assert len(hidden_sizes) > 0
        lstm_blocks: list[nn.Module] = []
        
        for hidden_size in hidden_sizes:
            lstm_blocks.extend(
                [
                    LSTMBlock(num_features, hidden_size, num_layers, bidirectional, dropout),
                    LSTMFCBlock(num_features),
                ]
            )

        self.lstm_blocks = nn.Sequential(*lstm_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.lstm_blocks(inputs)  # (T, N, num_features)
