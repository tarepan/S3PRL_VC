"""Network modules of Decoder"""


from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import MISSING


@dataclass
class ConfDecoderPreNet:
    """Configuration of Taco2AR Decoder PreNet.

    Args:
        dim_i - Dimension size of input
        dim_h_o - Dimension size of hidden and final layers
        n_layers - Number of FC layer
        dropout_rate: float=0.5    
    """
    dim_i: int = MISSING
    dim_h_o: int = MISSING
    n_layers: int = MISSING
    dropout_rate: float = MISSING

class Taco2Prenet(nn.Module):
    """Prenet module of Taco2AR Decoder.

    Model: (FC-ReLU-DO)xN | DO

    The Prenet preforms nonlinear conversion of inputs before input to auto-regressive lstm,
    which helps alleviate the exposure bias problem.

    Note:
        This module alway applies dropout even in evaluation.
        See the detail in `Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`.
    """

    def __init__(self, conf: ConfDecoderPreNet):
        super(Taco2Prenet, self).__init__()
        self._conf = conf
        self.dropout_rate = conf.dropout_rate

        # Dimension size check for DO only prenet (0-layers)
        if conf.n_layers == 0:
            assert dim_i == dim_h_o

        # WholeNet
        self.prenet = nn.ModuleList()
        for layer in range(conf.n_layers):
            n_inputs = conf.dim_i if layer == 0 else conf.dim_h_o
            self.prenet += [
                # FC-ReLU
                nn.Sequential(nn.Linear(n_inputs, conf.dim_h_o), nn.ReLU())
            ]

    def forward(self, x):
        # Make sure at least one dropout is applied even when there is no FC layer.
        if len(self.prenet) == 0:
            return F.dropout(x, self.dropout_rate)

        for i in range(len(self.prenet)):
            x = F.dropout(self.prenet[i](x), self.dropout_rate)
        return x


class ExLSTMCell(nn.Module):
    """Extended LSTM cell

    Model: input ---|
           z_t-1' ----LSTMCell---- z_t -[-LN][-DO][-FC-Tanh] - z_t'
           c_t-1  __|          |__ c_t
    """

    def __init__(self, dim_i: int, dim_h_o: int, dropout: float, layer_norm: bool, projection: bool):
        """
        Args:
            dim_i - Dimension size of input
            dim_h_o - Dimension size of LSTM hidden/cell state and total output
            dropout - Dropout probability
            layer_norm - Whether to use LayerNormalization
            projection - Whether to use non-linear projection after hidden state (LSTMP)
        """
        super().__init__()
        # Mode flags
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.proj = projection

        self.cell = nn.LSTMCell(dim_i, dim_h_o)

        # Normalization
        if self.layer_norm:
            self.ln = nn.LayerNorm(dim_h_o)

        # Dropout
        if self.dropout > 0:
            self.dp = nn.Dropout(p=dropout)

        # Projection: FC-Tanh
        if self.proj:
            self.pj = nn.Linear(dim_h_o, dim_h_o)

    def forward(self, input_x, z, c):
        """
        Step RNN cell.

        Args:
            input_x - RNN input_t
            z - hidden state t-1, could be projected hidden state
            c - cell state t-1
        Returns:
            new_z - hidden state t, could be projected, used also as output
            new_c - cell state t
        """

        new_z, new_c = self.cell(input_x, (z, c))

        if self.layer_norm:
            new_z = self.ln(new_z)

        if self.dropout > 0:
            new_z = self.dp(new_z)

        if self.proj:
            new_z = torch.tanh(self.pj(new_z))

        return new_z, new_c
