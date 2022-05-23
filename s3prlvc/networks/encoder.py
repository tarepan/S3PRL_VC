"""Network modules of Encoder"""


from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from extorch import Conv1dEx
from omegaconf import MISSING

# The follow section is related to Tacotron2
# Reference: https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/tacotron2

def encoder_init(m):
    """Initialize encoder parameters."""
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight, nn.init.calculate_gain("relu"))


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
        # store the hyperparameters
        self._conf = conf
        self.conv_residual = conf.conv_residual

        # segFC linear
        self.input_layer = nn.Linear(conf.dim_i, conf.conv_dim_c)

        # convs: [(Conv1d[-BN]-ReLU-DO)xN]
        if conf.num_conv_layers > 0:
            self.convs = nn.ModuleList()
            for layer in range(conf.num_conv_layers):
                ichans = conf.conv_dim_c
                # BN on/off, remaining is totally same
                ## BN (+)
                if conf.conv_batch_norm:
                    self.convs += [
                        nn.Sequential(
                            Conv1dEx(
                                ichans,
                                conf.conv_dim_c,
                                conf.conv_size_k,
                                stride=1,
                                padding=(conf.conv_size_k - 1) // 2,
                                bias=False,
                                causal=conf.causal,
                            ),
                            nn.BatchNorm1d(conf.conv_dim_c),
                            nn.ReLU(),
                            nn.Dropout(conf.conv_dropout_rate),
                        )
                    ]
                ## BN (-)
                else:
                    self.convs += [
                        nn.Sequential(
                            Conv1dEx(
                                ichans,
                                conf.conv_dim_c,
                                conf.conv_size_k,
                                stride=1,
                                padding=(conf.conv_size_k - 1) // 2,
                                bias=False,
                                causal=conf.causal,
                            ),
                            nn.ReLU(),
                            nn.Dropout(conf.conv_dropout_rate),
                        )
                    ]
        else:
            self.convs = None

        # blstm: [N-LSTM]
        if conf.num_rnn_layers > 0:
            iunits = conf.conv_dim_c if conf.num_conv_layers != 0 else embed_dim
            dim_lstm = conf.dim_o // 2 if conf.bidirectional else conf.dim_o
            self.blstm = nn.LSTM(
                iunits, dim_lstm, conf.num_rnn_layers, batch_first=True, bidirectional=conf.bidirectional
            )
            print(f"Encoder LSTM: {'bidi' if conf.bidirectional else 'uni'}")
        else:
            self.blstm = None

        # initialize
        self.apply(encoder_init)

    def forward(self, xs, ilens=None):
        """Calculate forward propagation.
        Args:
            xs (Batch, T_max, Feature_o): padded acoustic feature sequence
        """

        # segFC linear
        xs = self.input_layer(xs).transpose(1, 2)

        # Conv
        if self.convs is not None:
            for i in range(len(self.convs)):
                if self.conv_residual:
                    xs += self.convs[i](xs)
                else:
                    xs = self.convs[i](xs)

        # LSTM
        if self.blstm is None:
            return xs.transpose(1, 2)
        if not isinstance(ilens, torch.Tensor):
            ilens = torch.tensor(ilens)
        xs = pack_padded_sequence(xs.transpose(1, 2), ilens.cpu(), batch_first=True)
        self.blstm.flatten_parameters()
        xs, _ = self.blstm(xs)  # (B, Lmax, C)
        # Pack then Pad
        xs, hlens = pad_packed_sequence(xs, batch_first=True)
        return xs, hlens
