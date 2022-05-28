"""Network modules of Encoder"""


from dataclasses import dataclass
from typing import Any, List, Optional

from torch import nn, Tensor, tensor # pylint: disable=no-name-in-module
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from extorch import Conv1dEx # pyright: ignore [reportMissingTypeStubs]; bacause of extorch
from omegaconf import MISSING

# The follow section is related to Tacotron2
# Reference: https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/tacotron2

def encoder_init(module: Any):
    """Initialize encoder parameters."""
    if isinstance(module, nn.Conv1d):
        nn.init.xavier_uniform_(module.weight, nn.init.calculate_gain("relu")) # type: ignore ; caused by PyTorch


@dataclass
class ConfEncoder:
    """Configuration of Taco2AR Encoder

    Args:
        dim_i: Dimension of the inputs
        causal: Whether Conv1d is causal or not
        num_conv_layers: The number of conv layers
        conv_size_k: Size of conv kernel
        conv_dim_c: Dimension size of conv channels
        conv_batch_norm: Whether to use batch normalization for conv
        conv_residual: Whether to use residual connection for conv
        conv_dropout_rate: Conv dropout rate
        bidirectional: Whether RNN is bidirectional or not
        num_rnn_layers: The number of RNN layers
        dim_o: Dimension size of output, equal to RNN hidden size
    """
    dim_i: int = MISSING
    causal: bool = MISSING
    num_conv_layers: int = MISSING
    conv_dim_c: int = MISSING
    conv_size_k: int = MISSING
    conv_batch_norm: bool = MISSING
    conv_residual: bool = MISSING
    conv_dropout_rate: float = MISSING
    bidirectional: bool = MISSING
    num_rnn_layers: int = MISSING
    dim_o: int = MISSING

class Taco2Encoder(nn.Module):
    """Encoder module of Taco2AR.

    Model: segFC[-(Res(Conv1d[-BN]-ReLU-DO))xN][-LSTMxM]
    """

    def __init__(self, conf: ConfEncoder):
        super().__init__()

        # segFC linear
        self._seg_fc = nn.Linear(conf.dim_i, conf.conv_dim_c)

        # convs: [(Conv1d[-BN]-ReLU-DO)xN]
        self._use_conv_residual = conf.conv_residual
        self._convs = nn.ModuleList()
        for _ in range(conf.num_conv_layers):
            layer: List[nn.Module] = []
            layer += [Conv1dEx(
                        conf.conv_dim_c,
                        conf.conv_dim_c,
                        conf.conv_size_k,
                        stride=1,
                        padding=(conf.conv_size_k - 1) // 2,
                        bias=False,
                        causal=conf.causal,
            )]
            layer += [nn.BatchNorm1d(conf.conv_dim_c)] if conf.conv_batch_norm else []
            layer += [nn.ReLU()]
            layer += [nn.Dropout(conf.conv_dropout_rate)]
            self._convs += [nn.Sequential(*layer)]

        # blstm: [N-LSTM]
        if conf.num_rnn_layers > 0:
            dim_lstm = conf.dim_o // 2 if conf.bidirectional else conf.dim_o
            self.blstm = nn.LSTM(
                conf.conv_dim_c, dim_lstm, conf.num_rnn_layers, batch_first=True, bidirectional=conf.bidirectional
            )
            print(f"Encoder LSTM: {'bidi' if conf.bidirectional else 'uni'}")
        else:
            self.blstm = None

        # initialize
        self.apply(encoder_init)

    # Typing of PyTorch forward API is poor.
    def forward(self, x_series: Tensor, ilens: Optional[Tensor]=None): # pyright: reportIncompatibleMethodOverride=false
        """Calculate forward propagation.
        Args:
            x_series (Batch, T_max, Feature_o): padded acoustic feature sequence
        """

        # segFC linear
        x_series = self._seg_fc(x_series).transpose(1, 2)

        # Conv
        for conv_layer in self._convs:
            if self._use_conv_residual:
                x_series += conv_layer(x_series)
            else:
                x_series = conv_layer(x_series)

        # LSTM
        if self.blstm is None:
            return x_series.transpose(1, 2)
        if not isinstance(ilens, Tensor):
            ilens = tensor(ilens)
        xs_pack_seq = pack_padded_sequence(x_series.transpose(1, 2), ilens.cpu(), batch_first=True)
        self.blstm.flatten_parameters()
        xs_pack_seq, _ = self.blstm(xs_pack_seq)  # (B, Lmax, C)
        # Pack then Pad
        out_seq, hlens = pad_packed_sequence(xs_pack_seq, batch_first=True)
        return out_seq, hlens
