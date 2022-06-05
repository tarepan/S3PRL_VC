"""unit-to-mel Taco2AR network"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from torch import nn, Tensor, from_numpy, cat # pyright: ignore [reportUnknownVariableType] ; because of PyTorch; pylint: disable=no-name-in-module
from torch.nn.functional import interpolate # pyright: ignore [reportUnknownVariableType] ; because of PyTorch
from omegaconf import MISSING, SI

from .encoder import Taco2Encoder, ConfEncoder
from .conditioning import GlobalCondNet, ConfGlobalCondNet
from .decoder import Taco2Prenet, ConfDecoderPreNet, ExLSTMCell


@dataclass
class ConfDecoderMainNet:
    """
    Configuration of Taco2AR Decoder MainNet.

    Args:
        dim_i_cond - Dimension size of conditioning input
        dim_i_ar - Dimension size of processed AR input
        dim_h - Dimension size of RNN hidden units
        num_layers - The number of RNN layers
        dropout_rate - RNN dropout rate
        layer_norm - Whether to use layer normalization in RNN
        projection - Whether LSTM or LSTMP
        dim_o - Dimension size of output
    """
    dim_i_cond: int = MISSING
    dim_i_ar: int = MISSING
    dim_h: int = MISSING
    num_layers: int = MISSING
    dropout_rate: float = MISSING
    layer_norm: bool = MISSING
    projection: bool = MISSING
    dim_o: int = MISSING

@dataclass
class ConfTaco2ARNet:
    """
    Configuration of Taco2ARNet.

    Args:
        dim_latent - Dimension size of latent between Encoder and Decoder
        dim_processed_ar - Dimension size of processed Decoder AR feature
        dim_o - Dimension size of output acoustic feature
    """
    dim_latent: int = MISSING
    dim_processed_ar: int = MISSING
    dim_o: int  = MISSING
    encoder: ConfEncoder = ConfEncoder(
        dim_o=SI("${..dim_latent}"),)
    global_cond: ConfGlobalCondNet = ConfGlobalCondNet(
        dim_io=SI("${..dim_latent}"),)
    dec_prenet: ConfDecoderPreNet = ConfDecoderPreNet(
        dim_i=SI("${..dim_o}"),
        dim_h_o=SI("${..dim_processed_ar}"),)
    dec_mainnet: ConfDecoderMainNet = ConfDecoderMainNet(
        dim_i_cond=SI("${..dim_latent}"),
        dim_i_ar=SI("${..dim_processed_ar}"),
        dim_o=SI("${..dim_o}"))

class Taco2ARNet(nn.Module):
    """
    S3PRL-VC Taco2AR network.

    `Taco2-AR`: segFC-3Conv-1LSTM-cat_(z_t, AR-segFC)-NuniLSTM-segFC-segLinear
    segFC512-(Conv1d512_k5s1-BN-ReLU-DO_0.5)x3-1LSTM-cat_(z_t, AR-norm-(segFC-ReLU-DO)xN)-(1uniLSTM[-LN][-DO]-segFC-Tanh)xL-segFC
    """

    def __init__(self,
        resample_ratio: float,
        conf: ConfTaco2ARNet,
        mean: Optional[NDArray[np.float32]],
        scale: Optional[NDArray[np.float32]],
    ):
        """
        Args:
            resample_ratio - conditioning series resampling ratio
            conf - Configuration
            mean - FrequencyBand-wise mean
            scale - FrequencyBand-wise standard deviation
        """
        super().__init__()
        self._conf = conf
        self._resample_ratio = resample_ratio

        # Speaker-independent Encoder: segFC-Conv-LSTM // segFC512-(Conv1d512_k5s1-BN-ReLU-DO_0.5)x3-1LSTM
        self.encoder = Taco2Encoder(conf.encoder)

        # Global speaker conditioning network
        self.global_cond = GlobalCondNet(conf.global_cond)

        # Decoder
        ## PreNet: (segFC-ReLU-DO)xN
        if ((mean is None) and (scale is not None)) or ((mean is not None) and (scale is None)):
            raise Exception("Should be 'both mean/scale exist' OR 'both mean/scale not exist'")
        elif (mean is not None) and (scale is not None):
            self.register_spec_stat(mean, scale)
        self.prenet = Taco2Prenet(conf.dec_prenet)
        ## MainNet: LSTMP + linear projection
        conf_mainnet = conf.dec_mainnet
        ### LSTMP: (1uniLSTM[-LN][-DO][-segFC-Tanh])xN
        self.lstmps = nn.ModuleList()
        for i_layer in range(conf_mainnet.num_layers):
            # cat(local_cond, process(ar)) OR lower layer hidden_state/output
            dim_i_lstm = conf_mainnet.dim_i_cond + conf_mainnet.dim_i_ar if i_layer == 0 else conf_mainnet.dim_h
            rnn_layer = ExLSTMCell(
                dim_i=dim_i_lstm,
                dim_h_o=conf_mainnet.dim_h,
                dropout=conf_mainnet.dropout_rate,
                layer_norm=conf_mainnet.layer_norm,
                projection=conf_mainnet.projection,
            )
            self.lstmps.append(rnn_layer)
        ### Projection: segFC
        self.proj = nn.Linear(conf_mainnet.dim_h, conf_mainnet.dim_o)
        self._dim_o = conf_mainnet.dim_o
        ## PostNet: None

    def register_spec_stat(self, mean: NDArray[np.float32], scale: NDArray[np.float32]) -> None:
        """
        Register spectram statistics as model state.

        Args:
            mean -  frequencyBand-wise mean
            scale - frequencyBand-wise standard deviation
        """
        # buffer is part of state_dict (saved by PyTorch functions)
        self.register_buffer("target_mean", from_numpy(mean).float())
        self.register_buffer("target_scale", from_numpy(scale).float())

    # Typing of PyTorch forward API is poor.
    def forward(self, # pyright: reportIncompatibleMethodOverride=false
        features: Tensor,
        lens: Tensor,
        spk_emb: Tensor,
        targets: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Convert unit sequence into acoustic feature sequence.

        Args:
            features (Batch, T_max, Feature_i): input unit sequences
            lens (Batch) - Lengths of unpadded input unit sequence
            spk_emb (Batch, Spk_emb): speaker embedding vectors as global conditioning
            targets (Batch, T_max, Feature_o): padded target acoustic feature sequences
        Returns:
            ((Batch, Tmax, Freq), lens)
        """
        batch = features.shape[0]

        # Resampling: resample the input features according to resample_ratio
        # (B, T_max, Feat_i) => (B, Feat_i, T_max) => (B, Feat_i, T_max') => (B, T_max', Feat_i)
        features = features.permute(0, 2, 1)
        # Nearest interpolation
        resampled_features: Tensor = interpolate(features, scale_factor = self._resample_ratio)
        resampled_features = resampled_features.permute(0, 2, 1)
        lens = lens * self._resample_ratio

        # (resampled_features:(B, T_max', Feat_i)) -> (B, T_max', Feat_h)
        # `lens` is used for RNN padding. `si` stands for speaker-independent
        si_latent_series, lens = self.encoder(resampled_features, lens)

        # (B, T_max', Feat_h) -> (B, T_max', Feat_h)
        conditioning_series = self.global_cond(si_latent_series, spk_emb)

        # Decoder: spec_t' = f(spec_t, cond_t), cond_t == f(unit_t, spk_g)
        # AR decofing w/ or w/o teacher-forcing
        # Transpose for easy access: (B, T_max, Feat_o) => (T_max, B, Feat_o)
        if targets is not None:
            targets = targets.transpose(0, 1)
        predicted_list: List[Tensor] = []
        # Initialize LSTM hidden state and cell state of all LSTMP layers, and x_t-1
        c_list: List[Tensor] = []
        z_list: List[Tensor] = []
        _tensor = conditioning_series
        for _ in range(0, len(self.lstmps)):
            c_list += [_tensor.new_zeros(batch, self._conf.dec_mainnet.dim_h)]
            z_list += [_tensor.new_zeros(batch, self._conf.dec_mainnet.dim_h)]
        prev_out = _tensor.new_zeros(batch, self._conf.dec_mainnet.dim_o)

        # step-by-step loop for autoregressive decoding
        ## local_cond::(B, hidden_dim)
        for step, local_cond in enumerate(conditioning_series.transpose(0, 1)):
            # Single time step
            ## RNN input (local conditioning and processed AR)
            i_ar = self.prenet(prev_out)
            cond_plus_ar = cat([local_cond, i_ar], dim=1)
            ## Run single time step of all LSTMP layers
            for i, lstmp in enumerate(self.lstmps):
                # Run a layer (1uniLSTM[-LN][-DO]-segFC-Tanh), then update states
                # Input: RNN input OR lower layer's output
                lstmp_input = cond_plus_ar if i == 0 else z_list[i-1]
                z_list[i], c_list[i] = lstmp(lstmp_input, z_list[i], c_list[i])
            # Projection & Stack: Stack output_t `proj(o_lstmps)` in full-time list
            predicted_list += [self.proj(z_list[-1]).view(batch, self._dim_o, -1)]
            # teacher-forcing if `target` else pure-autoregressive
            prev_out = targets[step] if targets is not None else predicted_list[-1].squeeze(-1)
            # AR spectrum is normalized (todo: could be moved up, but it change t=0 behavior)
            prev_out = (prev_out - self.target_mean) / self.target_scale
            # /Single time step
        # (Batch, Freq, 1?)[] -> (Batch, Freq, Tmax) -> (Batch, Tmax, Freq)
        predicted = cat(predicted_list, dim=2).transpose(1, 2)

        return predicted, lens
