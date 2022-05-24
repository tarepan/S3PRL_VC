"""Taco2AR model"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from torch import Tensor, tensor, from_numpy, maximum # pylint: disable=no-name-in-module
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import MISSING
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from resemblyzer import preprocess_wav, VoiceEncoder

from .networks.taco2ar import Taco2ARNet, ConfTaco2ARNet
from .data.dataset import Stat
from .utils import make_non_pad_mask


class Loss(nn.Module):
    """
    L1 loss module supporting (1) loss calculation in the normalized target feature space
                              (2) masked loss calculation
    """
    def __init__(self, stats: Optional[Stat]):
        """
        Args:
            stats: Mean and Scale statistics for normalization
        """
        super().__init__()
        self.objective = nn.L1Loss(reduction="mean")
        if stats:
            self._register_spec_stat(stats)

    def _register_spec_stat(self, stats: Stat):
        """
        Register spectram statistics as model state.

        Args:
            stats
                .mean_::np.ndarray -  frequencyBand-wise mean 
                .scale_::np.ndarray - frequencyBand-wise standard deviation
        """
        # buffer is part of state_dict (saved by PyTorch functions)
        self.register_buffer("target_mean", from_numpy(stats.mean_).float())
        self.register_buffer("target_scale", from_numpy(stats.scale_).float())

    def normalize(self, x):
        return (x - self.target_mean) / self.target_scale

    def forward(self, x, y, x_lens, y_lens, device):
        """
        Args:
            x::Tensor[Batch, Tmax, Freq] - predicted_features
            y - acoustic_features_padded
            x_lens - predicted_feature_lengths
            y_lens - acoustic_feature_lengths
            device
        """
        # match the input feature length to acoustic feature length to calculate the loss
        if x.shape[1] > y.shape[1]:
            x = x[:, :y.shape[1]]
            masks = make_non_pad_mask(y_lens).unsqueeze(-1).to(device)
        if x.shape[1] <= y.shape[1]:
            y = y[:, :x.shape[1]]
            masks = make_non_pad_mask(x_lens).unsqueeze(-1).to(device)        

        x_normalized = self.normalize(x)
        y_normalized = self.normalize(y.to(device))

        # slice based on mask by PyTorch function
        x_masked = x_normalized.masked_select(masks)
        y_masked = y_normalized.masked_select(masks)

        loss = self.objective(x_masked, y_masked)
        return loss


@dataclass
class ConfOptim:
    """Configuration of optimizer.
    Args:
        learning_rate: Optimizer learning rate
        sched_warmup_step: The number of LR shaduler warmup steps
        sched_total_step: The number of total training steps
    """
    learning_rate: float = MISSING
    sched_warmup_step: int = MISSING
    sched_total_step: int = MISSING

@dataclass
class ConfTaco2ARVC:
    """
    Args:
        expdir - Directory in which dev/test results are saved
    """
    sr_for_unit: int = MISSING
    sr_for_mel: int = MISSING
    unit_hop_length: int = MISSING
    mel_hop_length: int = MISSING
    net: ConfTaco2ARNet = ConfTaco2ARNet()
    optim: ConfOptim = ConfOptim(
        sched_total_step="${..train_steps}",)

class Taco2ARVC(pl.LightningModule):
    """Taco2AR unit-to-mel VC model.
    """

    def __init__(self, conf: ConfTaco2ARVC, stats: Optional[Stat]):
        super().__init__()
        self.save_hyperparameters()
        self._conf = conf

        # Time-directional up/down sampling ratio toward input series
        ## Upstream:   [sample/sec] / [sample/unit]     = [unit/sec]
        unit_per_sec = conf.sr_for_unit / conf.unit_hop_length
        ## Downstream: [sample/sec] / [sample/melFrame] = [melFrame/sec]
        mel_per_sec = conf.sr_for_mel / conf.mel_hop_length
        resample_ratio = mel_per_sec / unit_per_sec

        # define model and loss
        self.model = Taco2ARNet(
            resample_ratio=resample_ratio,
            conf=conf.net,
            mean=stats.mean_ if stats else None,
            scale=stats.scale_ if stats else None,
        )
        self.objective = Loss(stats)

        # Utterance embedding model for inference
        self.uttr_encoder = None

    def forward(self, # pylint: disable=arguments-differ
                split,
                input_features,
                acoustic_features,
                acoustic_features_padded,
                acoustic_feature_lengths,
                spk_embs,
                vc_ids,
                records):
        """(PL API) Forward a batch.

        Args:
            split: mode
            input_features: list of unpadded features generated by the upstream
            acoustic_features: List[Tensor(`lmspc`)], not used...?
            acoustic_features_padded: `acoustic_features` padded by PyTorch function
            acoustic_feature_lengths: Tensor(feature time length)
            spk_embs: Tensor(`ref_spk_emb`)
            vc_ids: List[(target_spk, source_spk, uttr_name)]
        """
        pass

    # Typing of PL step API is poor. It is typed as `(self, *args, **kwargs)`.
    def training_step(self, batch): # pylint: disable=arguments-differ
        """(PL API) Forward a batch.

        Args:
            batch
                input_features
                input_feature_lengths
                acoustic_features_padded
                acoustic_feature_lengths
                spk_embs - Speaker embeddings
                device - device
            batch_idx - Batch index in a training epoch
        Returns - loss
        """
        input_features, input_feature_lengths, \
            acoustic_features_padded, acoustic_feature_lengths, \
            spk_embs, device = batch

        # The forward
        predicted_features, predicted_feature_lengths = self.model(
            input_features, input_feature_lengths, \
            spk_embs,
            acoustic_features_padded,
        )

        # Masked/normalized L1 loss
        loss = self.objective(predicted_features,
                              acoustic_features_padded,
                              predicted_feature_lengths,
                              acoustic_feature_lengths,
                              device)

        self.log("loss", loss)
        return {"loss": loss}

    def validation_step(self, batch): # pylint: disable=arguments-differ
        """(PL API) Validate a batch.
        """

        input_features, input_feature_lengths, \
            acoustic_features_padded, acoustic_feature_lengths, \
            spk_embs, device = batch

        predicted_features, predicted_feature_lengths = self.model(
            input_features,
            input_feature_lengths,
            spk_embs,
        )
        # Masked/normalized L1 loss
        loss = self.objective(predicted_features,
                            acoustic_features_padded,
                            predicted_feature_lengths,
                            acoustic_feature_lengths,
                            device)
        self.log("val_loss", loss)

        # todo: Synthesis
        # [PyTorch](https://pytorch.org/docs/stable/tensorboard.html#torch.
        #     utils.tensorboard.writer.SummaryWriter.add_audio)
        # self.logger.experiment.add_audio(
        #     f"audio_{batch_idx}",
        #     wave, # snd_tensor: Tensor(1, L)
        #     global_step=self.global_step,
        #     sample_rate=self._conf.sampling_rate,
        # )

        # return anything_for_`validation_epoch_end`

    # def test_step(self, batch, batch_idx: int):
    #     """(PL API) Test a batch. If not provided, test_step == validation_step."""
    #     return anything_for_`test_epoch_end`

    def predict_step(self, batch): # pylint: disable=arguments-differ
        """(PL API) Generate a mel-spectrogram from a unit sequence and speaker embedding.
        Args:
            batch
                unit_series::Tensor[Batch==1, TimeUnit, Feat] - Input unit sequence
                target_emb::Tensor[Batch==1, Emb] - Target style embedding
        Returns:
            Tensor[Batch==1, TimeMel, Freq] - mel-spectrogram
        """
        unit_series, target_emb = batch
        return self.model(unit_series.to(self.device), target_emb.to(self.device))

    def configure_optimizers(self):
        """Set up a optimizer
        """

        optim = AdamW(self.model.parameters(), lr=self._conf.optim.learning_rate)

        # Scheduler's multiplicative factor function
        total_steps = self._conf.optim.sched_total_step
        warmup_steps = self._conf.optim.sched_warmup_step
        def lr_lambda(now: int) -> float:
            """0@0 ---> (linear) ---> 1@`warmup_steps` ---> (linear) ---> 0@`total_steps`"""
            is_warmup = now < warmup_steps
            increasing = now / warmup_steps
            decreasing = (total_steps - now) / (total_steps - warmup_steps)
            return increasing if is_warmup else decreasing

        sched = {
            "scheduler": LambdaLR(optim, lr_lambda),
            "interval": "step",
        }

        return {
            "optimizer": optim,
            "lr_scheduler": sched,
        }

    def mel_taco_to_rnnms(self, log_amp_bel: Tensor) -> Tensor:
        """
        Convert TacoVC-compatible mel-spectrogram to RNNMS-compatible one.

        Args:
            log_amp_bel::[Batch==1, TimeMel, Freq] - log(ref=0dB, min=-200dB)-amplitude [B]
        Returns:
            rnnms_mel::[Batch==1, TimeMel, Freq] - scaled(1/80)-log(ref=20dB, minrel=-80dB)-power
        """
        # log(ref=0dB, min=-200dB)-amplitude [dB]
        log_amp_db = 10. * log_amp_bel
        # log(S^2/1) = 2*log(S/1) ==> 10*log(S^2/1) [dB] = 10*2*log(S/1) = 2*(10*log(S/1))
        log_pow = 2. * log_amp_db
        log_pow_ref20 = log_pow - 20.
        log_pow_ref20_minrel80 = maximum(tensor([-80.]), log_pow_ref20)
        return log_pow_ref20_minrel80 / 80.

    def wavs2emb(self, waves: List[np.ndarray]) -> Tensor:
        """Convert waveforms into an averaged embedding.

        Args:
            waves::List[(Time,)] - waveforms, each of which can have different length
        Returns:
            ave_emb::Tensor[Batch=1, Emb] - an averaged embedding
        """

        # Initialization at first call
        if self.uttr_encoder is None:
            self.uttr_encoder = VoiceEncoder().to(self.device)

        # Calculate an average of utterance embeddings
        processed_waves = [preprocess_wav(wave) for wave in waves]
        ave_emb = self.uttr_encoder.embed_speaker(processed_waves)

        return from_numpy(ave_emb).unsqueeze(dim=0)
