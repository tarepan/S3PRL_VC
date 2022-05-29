"""Network modules of Encoder"""


from dataclasses import dataclass, asdict
from typing import Any, List, Optional

from torch import nn, Tensor, tensor # pylint: disable=no-name-in-module
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
    in_channels: int = MISSING # conf.conv_dim_c
    out_channels: int = MISSING # conf.conv_dim_c
    kernel_size: int = MISSING # conf.conv_size_k
    stride=1
    bias=False
    causal: bool = MISSING # conf.causal

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
        self._seg_fc = nn.Linear(conf.dim_i, conf.dim_h)

        # Convs: (Conv1d[-BN]-ReLU-DO)xN
        self._use_conv_residual = conf.conv_residual
        self._convs = nn.ModuleList()
        for _ in range(conf.num_conv_layers):
            layer: List[nn.Module] = []
            layer += [Conv1dEx(padding=(conf.conv.kernel_size - 1) // 2, **asdict(conf.conv))]
            layer += [nn.BatchNorm1d(conf.conv.out_channels)] if conf.conv_batch_norm else []
            layer += [nn.ReLU()]
            layer += [nn.Dropout(conf.conv_dropout_rate)]
            self._convs += [nn.Sequential(*layer)]
        self.apply(encoder_init)

        # RNN: N-LSTM
        self._use_rnn = conf.rnn.num_layers > 0
        if self._use_rnn:
            dim_lstm_h = conf.dim_o // 2 if conf.rnn.bidirectional else conf.dim_o
            self._rnn = nn.LSTM(hidden_size=dim_lstm_h, **asdict(conf.rnn))

    # Typing of PyTorch forward API is poor.
    def forward(self, x_series: Tensor, ilens: Optional[Tensor]=None): # pyright: reportIncompatibleMethodOverride=false
        """Calculate forward propagation.
        Args:
            x_series (Batch, T_max, Feature_o): padded acoustic feature sequence
        """

        # SegFC
        x_series = self._seg_fc(x_series).transpose(1, 2)

        # Conv
        for conv_layer in self._convs:
            if self._use_conv_residual:
                x_series += conv_layer(x_series)
            else:
                x_series = conv_layer(x_series)

        # Early return w/o LSTM
        if self._use_rnn is False:
            return x_series.transpose(1, 2)

        # RNN
        if not isinstance(ilens, Tensor):
            ilens = tensor(ilens)
        xs_pack_seq = pack_padded_sequence(x_series.transpose(1, 2), ilens.cpu(), batch_first=True)
        self._rnn.flatten_parameters()
        xs_pack_seq, _ = self._rnn(xs_pack_seq)  # (B, Lmax, C)
        # Pack then Pad
        out_seq, hlens = pad_packed_sequence(xs_pack_seq, batch_first=True)
        return out_seq, hlens
