"""Network modules of Encoder"""


from dataclasses import dataclass, asdict
from typing import Any, List, Tuple

from torch import nn, Tensor # pylint: disable=no-name-in-module
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from extorch import Conv1dEx # pyright: ignore [reportMissingTypeStubs]; bacause of extorch
from omegaconf import MISSING, SI


def encoder_init(module: Any):
    """Initialize encoder parameters."""
    if isinstance(module, nn.Conv1d):
        nn.init.xavier_uniform_(module.weight, nn.init.calculate_gain("relu")) # type: ignore ; caused by PyTorch


@dataclass
class ConfConv:
    """
    Configuration of Conv (Ex), directly unpackable.

    Args:
        in_channels - Channel size of input
        out_channels - Channel size of output
        kernel_size - Size of conv kernel
        causal - Whether Conv1d is causal or not
    """
    in_channels: int = MISSING
    out_channels: int = MISSING
    kernel_size: int = MISSING
    stride: int = 1
    bias: bool = False
    causal: bool = MISSING

@dataclass
class ConfRNN:
    """
    Configuration of nn.RNN/GRU/LSTM, directly unpackable.

    Args:
        input_size - Dimension size of RNN input
        num_layers - The number of RNN layers
        bidirectional - Whether RNN is bidirectional or not
    """
    input_size: int = MISSING
    num_layers: int = MISSING
    batch_first: bool = True
    bidirectional: bool = MISSING

@dataclass
class ConfEncoder:
    """Configuration of Taco2AR Encoder

    Args:
        dim_i: Dimension of the inputs
        dim_h: Dimension of hidden layer, equal to o_segFC, i_conv, o_conv and i_rnn
        num_conv_layers: The number of conv layers
        conv_batch_norm: Whether to use batch normalization for conv
        conv_dropout_rate: Conv dropout rate
        conv_residual: Whether to use residual connection for conv
        dim_o: Dimension size of output, equal to RNN hidden size
    """
    dim_i: int = MISSING
    dim_h: int = MISSING
    num_conv_layers: int = MISSING
    conv: ConfConv = ConfConv(
        in_channels=SI("${..dim_h}"),
        out_channels=SI("${..dim_h}"),)
    conv_batch_norm: bool = MISSING
    conv_dropout_rate: float = MISSING
    conv_residual: bool = MISSING
    rnn: ConfRNN = ConfRNN(
        input_size=SI("${..dim_h}"),)
    dim_o: int = MISSING

class Taco2Encoder(nn.Module):
    """Taco2AR's Encoder network.

    Model: <series>-segFC-([Res](Conv1d[-BN]-ReLU-DO))xN-LSTMxM-<series>
    """

    def __init__(self, conf: ConfEncoder):
        super().__init__()

        # SegFC: Linear
        self.seg_fc = nn.Linear(conf.dim_i, conf.dim_h)

        # Convs: (Conv1d[-BN]-ReLU-DO)xN
        self._use_conv_residual = conf.conv_residual
        self.convs = nn.ModuleList()
        for _ in range(conf.num_conv_layers):
            layer: List[nn.Module] = []
            layer += [Conv1dEx(padding=(conf.conv.kernel_size - 1) // 2, **asdict(conf.conv))]
            layer += [nn.BatchNorm1d(conf.conv.out_channels)] if conf.conv_batch_norm else []
            layer += [nn.ReLU()]
            layer += [nn.Dropout(conf.conv_dropout_rate)]
            self.convs += [nn.Sequential(*layer)]
        self.apply(encoder_init)

        # RNN: N-LSTM
        self._use_rnn = conf.rnn.num_layers > 0
        if self._use_rnn:
            dim_lstm_h = conf.dim_o // 2 if conf.rnn.bidirectional else conf.dim_o
            self.rnn = nn.LSTM(hidden_size=dim_lstm_h, **asdict(conf.rnn))

    # Typing of PyTorch forward API is poor.
    def forward(self, x_series_padded: Tensor, len_x_series: Tensor) -> Tuple[Tensor, Tensor]: # pyright: reportIncompatibleMethodOverride=false
        """Calculate forward.

        Args:
            x_series_padded :: (Batch, T_max, Feat_i) - Padded input series
            len_x_series    :: (Batch)                - Lengths of each input series
        Returns:
            o_series_padded :: (Batch, T_max, Feat_o) - Padded output series
            len_o_series    :: (Batch)                - Lengths of each output series
        """

        # SegFC :: (B, Tmax, Feat_i) -> (B, Tmax, Feat_h)
        x_series_padded = self.seg_fc(x_series_padded)

        # Conv :: (B, Tmax, Feat_h) -> (B, Feat_h, Tmax) -> (B, Feat_h, Tmax)
        x_series_padded = x_series_padded.transpose(1, 2)
        for conv_layer in self.convs:
            if self._use_conv_residual:
                x_series_padded += conv_layer(x_series_padded)
            else:
                x_series_padded = conv_layer(x_series_padded)

        # Early return w/o LSTM :: (B, Feat_h, Tmax) -> (B, Tmax, Feat_h)
        if self._use_rnn is False:
            # Current config use Feat_h == Feat_o, so consistent, but other non-equal configs cause size inconsistency.
            return x_series_padded.transpose(1, 2), len_x_series

        # RNN
        ## Packing :: (B, Feat_h, Tmax) -> (B, Tmax, Feat_h) -> PackedSequence(B, Ti, Feat_h)
        x_series_packed = pack_padded_sequence(x_series_padded.transpose(1, 2), len_x_series.cpu(), batch_first=True)
        ## RNN :: PackedSequence(B, Ti, Feat_h) -> PackedSequence(B, Ti, Feat_o)
        self.rnn.flatten_parameters()
        x_series_packed, _ = self.rnn(x_series_packed)
        ## UnPacking w/ padding :: PackedSequence(B, Ti, Feat_o) -> (B, Tmax, Feat_o)
        o_series_padded, len_o_series = pad_packed_sequence(x_series_packed, batch_first=True)

        return o_series_padded, len_o_series
