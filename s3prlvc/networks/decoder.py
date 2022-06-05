"""Network modules of Decoder"""


from dataclasses import dataclass
from typing import List, Tuple

from torch import nn, Tensor # pylint: disable=no-name-in-module
import torch.nn.functional as F

from omegaconf import MISSING


class AlwaysDropout(nn.Module):
    """
    Special Dropout module, which is applied even in evaluation mode.

    PyTorch nn.Dropout: https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
    PyTorch  F.dropout: https://pytorch.org/docs/stable/generated/torch.nn.functional.dropout.html
    """
    def __init__(self, prob: float):
        super().__init__()
        self._prob = prob
    def forward(self, inputs: Tensor) -> Tensor:
        """Dropout always."""
        return F.dropout(inputs, self._prob, training=True)


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
        super().__init__()

        layers: List[nn.Module] = []
        for layer in range(conf.n_layers):
            # Single layer: SegFC-ReLU-DO
            dim_i_layer = conf.dim_i if layer == 0 else conf.dim_h_o
            layers += [
                nn.Linear(dim_i_layer, conf.dim_h_o),
                nn.ReLU(),
                AlwaysDropout(conf.dropout_rate),
            ]
        # Make sure at least one dropout is applied even when there is no FC layer.
        if conf.n_layers == 0:
            assert conf.dim_i == conf.dim_h_o
            layers += [AlwaysDropout(conf.dropout_rate)]

        self.net = nn.Sequential(*layers)

    # Typing of PyTorch forward API is poor.
    def forward(self, x_step: Tensor) -> Tensor: # pyright: reportIncompatibleMethodOverride=false
        """
        Args:
            x_step - single step input
        """
        return self.net(x_step)


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

        # Ex post processings
        posts: List[nn.Module] = []
        posts += [nn.LayerNorm(dim_h_o)] if layer_norm else []
        posts += [nn.Dropout(p=dropout)] if dropout > 0 else []
        posts += [nn.Linear(dim_h_o, dim_h_o), nn.Tanh()] if projection else []
        self.posts = nn.Sequential(*posts)

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
        new_z = self.posts(new_z)
        return new_z, new_c
