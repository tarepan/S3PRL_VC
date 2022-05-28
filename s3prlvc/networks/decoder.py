"""Network modules of Decoder"""


from dataclasses import dataclass
from typing import Tuple

from torch import nn, Tensor, tanh # pylint: disable=no-name-in-module
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
        See the detail in
        `Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`.
    """

    def __init__(self, conf: ConfDecoderPreNet):
        super(Taco2Prenet, self).__init__()
        self._conf = conf
        self.dropout_rate = conf.dropout_rate

        # Dimension size check for DO only prenet (0-layers)
        if conf.n_layers == 0:
            assert conf.dim_i == conf.dim_h_o

        # WholeNet
        self.prenet = nn.ModuleList()
        for layer in range(conf.n_layers):
            n_inputs = conf.dim_i if layer == 0 else conf.dim_h_o
            self.prenet += [
                # FC-ReLU
                nn.Sequential(nn.Linear(n_inputs, conf.dim_h_o), nn.ReLU())
            ]

    # Typing of PyTorch forward API is poor.
    def forward(self, x_step: Tensor) -> Tensor: # pyright: reportIncompatibleMethodOverride=false
        """
        Args:
            x_step - single step input
        """
        # Make sure at least one dropout is applied even when there is no FC layer.
        if len(self.prenet) == 0:
            return F.dropout(x_step, self.dropout_rate)

        for i in range(len(self.prenet)):
            x_step = F.dropout(self.prenet[i](x_step), self.dropout_rate)
        return x_step


class ExLSTMCell(nn.Module):
    """Extended LSTM cell

    Model: input ---|
           z_t-1' ----LSTMCell---- z_t -[-LN][-DO][-FC-Tanh] - z_t'
           c_t-1  __|          |__ c_t
    """

    def __init__(self,
        dim_i: int,
        dim_h_o: int,
        dropout: float,
        layer_norm: bool,
        projection: bool
    ):
        """
        Args:
            dim_i - Dimension size of input
            dim_h_o - Dimension size of LSTM hidden/cell state and total output
            dropout - Dropout probability
            layer_norm - Whether to use LayerNormalization
            projection - Whether to use non-linear projection after hidden state (LSTMP)
        """
        super().__init__()

        # Common LSTM cell
        self.cell = nn.LSTMCell(dim_i, dim_h_o)

        # Modes
        ## Normalization
        self._use_layer_norm = layer_norm
        if self._use_layer_norm:
            self.layer_norm = nn.LayerNorm(dim_h_o)
        ## Dropout
        self._use_dropout = dropout > 0
        if self._use_dropout:
            self.dropout = nn.Dropout(p=dropout)
        ## Projection
        self._use_projection = projection
        if self._use_projection:
            self.projection = nn.Linear(dim_h_o, dim_h_o)

    # Typing of PyTorch forward API is poor.
    def forward(self, # pyright: reportIncompatibleMethodOverride=false
        input_x: Tensor,
        hidden_state: Tensor,
        cell_state: Tensor
    ) -> Tuple[Tensor, Tensor]:
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

        new_z, new_c = self.cell(input_x, (hidden_state, cell_state))

        if self._use_layer_norm:
            new_z = self.layer_norm(new_z)

        if self._use_dropout:
            new_z = self.dropout(new_z)

        if self._use_projection:
            new_z = tanh(self.projection(new_z))

        return new_z, new_c
