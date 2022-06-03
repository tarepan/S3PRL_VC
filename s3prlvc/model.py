"""Taco2AR model"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from torch import nn, Tensor, tensor, from_numpy, maximum, device as PTdevice # pyright: ignore [reportUnknownVariableType]; pylint: disable=no-name-in-module ; bacause of PyTorch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
from omegaconf import MISSING, SI
from resemblyzer import preprocess_wav, VoiceEncoder # pyright: ignore [reportUnknownVariableType, reportMissingTypeStubs]; bacause of resemblyzer
from parallel_wavegan.runners import HiFiGAN # pyright: ignore [reportMissingTypeStubs]; bacause of resemblyzeparallel_wavegan

from .networks.taco2ar import Taco2ARNet, ConfTaco2ARNet
from .data.dataset import Stat
from .utils import make_non_pad_mask # pyright: ignore [reportUnknownVariableType];


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
                .mean_ -  frequencyBand-wise mean
                .scale_ - frequencyBand-wise standard deviation
        """
        # buffer is part of state_dict (saved by PyTorch functions)
        self.register_buffer("target_mean", from_numpy(stats.mean_).float())
        self.register_buffer("target_scale", from_numpy(stats.scale_).float())

    def normalize(self, x_series: Tensor) -> Tensor:
        """Normalize input with pre-registered statistics."""
        return (x_series - self.target_mean) / self.target_scale

    # Typing of PyTorch forward API is poor.
    def forward(self, # pyright: ignore [reportIncompatibleMethodOverride]
        prediction_padded: Tensor,
        target_padded: Tensor,
        x_lens: Tensor,
        y_lens: Tensor,
        device: PTdevice,
    ) -> Tensor:
        """
        Args:
            prediction_padded::(Batch, Tmax, Freq) - predicted mel-spectrogram, padded
            target_padded::(Batch, Tmax, Freq) - target mel-spectrogram, padded
            x_lens::(Batch) - Lengthes of non-padded predicted mel-spectrogram
            y_lens::(Batch) - Lengthes of non-padded target mel-spectrogram
            device
        """
        # match the input feature length to acoustic feature length to calculate the loss
        if prediction_padded.shape[1] > target_padded.shape[1]:
            prediction_padded = prediction_padded[:, :target_padded.shape[1]]
            masks = make_non_pad_mask(y_lens).unsqueeze(-1).to(device)
        if prediction_padded.shape[1] <= target_padded.shape[1]:
            target_padded = target_padded[:, :prediction_padded.shape[1]]
            masks = make_non_pad_mask(x_lens).unsqueeze(-1).to(device)

        x_normalized = self.normalize(prediction_padded)
        y_normalized = self.normalize(target_padded.to(device))

        # slice based on mask by PyTorch function
        x_masked = x_normalized.masked_select(masks)
        y_masked = y_normalized.masked_select(masks)

        loss = self.objective(x_masked, y_masked)
        return loss


@dataclass
class ConfMel2Wav:
    """Configuration of mel2wav (vocoder).
    Args:
        path_state - Path to the mel2wav vocoder state (checkpoint)
    """
    path_state: str = MISSING

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
        sched_total_step=SI("${..train_steps}"),)
    mel2wav: ConfMel2Wav = ConfMel2Wav()

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
        self.network = Taco2ARNet(
            resample_ratio=resample_ratio,
            conf=conf.net,
            mean=stats.mean_ if stats else None,
            scale=stats.scale_ if stats else None,
        )
        self.objective = Loss(stats)

        # Utterance embedding model for inference
        self.uttr_encoder = None

        # Vocoder for mel2wav
        # url_ckpt = "https://drive.google.com/file/d/12w1LpF6HjsJBmOUUkS6LV1d7AX18SA7u"
        # download_pretrained_model(url_ckpt, download_dir=None)
        self._vocoder = HiFiGAN(conf.mel2wav.path_state)

    # def forward(self, # pylint: disable=arguments-differ
    #             split: str,
    #             input_features,
    #             acoustic_features,
    #             acoustic_features_padded: List[Tensor],
    #             acoustic_feature_lengths: Tensor,
    #             spk_embs: Tensor,
    #             vc_ids,
    #             records):
    #     """(PL API) Forward a batch.

    #     Args:
    #         split: mode
    #         input_features: list of unpadded features generated by the upstream
    #         acoustic_features: List[Tensor(`lmspc`)], not used...?
    #         acoustic_features_padded: `acoustic_features` padded by PyTorch function
    #         acoustic_feature_lengths: Tensor(feature time length)
    #         spk_embs: Tensor(`ref_spk_emb`)
    #         vc_ids: List[(target_spk, source_spk, uttr_name)]
    #     """
    #     pass

    # Typing of PL step API is poor. It is typed as `(self, *args, **kwargs)`.
    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]): # pyright: ignore [reportIncompatibleMethodOverride] ; pylint: disable=arguments-differ
        """(PL API) Forward a batch.

        Args:
            batch
                unit_series_padded - Padded input unit series
                len_unit_series - Length of non-paded unit_series
                mspc_series_padded - Padded target melspectrogram
                len_mspc_series - Length of non-padded melspectrograms
                spk_embs - Speaker embeddings
                vc_ids - Voice conversion source/target identities
        """
        unit_series_padded, len_unit_series, mspc_series_padded, len_mspc_series, spk_embs, _ = batch

        # The forward with AR teacher-forcing
        mspc_series_padded_predicted, len_mspc_series_predicted = self.network(
            unit_series_padded,
            len_unit_series,
            spk_embs,
            mspc_series_padded,
        )

        # Masked/normalized L1 loss
        loss = self.objective(
            mspc_series_padded_predicted,
            mspc_series_padded,
            len_mspc_series_predicted,
            len_mspc_series,
            self.device
        )

        self.log("loss", loss) #type: ignore ; because of PyTorch-Lightning
        return {"loss": loss}

    def validation_step(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor], _: int): # pyright: ignore [reportIncompatibleMethodOverride] ; pylint: disable=arguments-differ
        """(PL API) Validate a batch.

        Args:
            batch
                unit_series_padded - Padded input unit series
                len_unit_series - Length of non-paded unit_series
                melspec_padded - Padded target melspectrogram
                len_melspec - Length of non-padded melspectrograms
                spk_embs - Speaker embeddings
                vc_ids - Voice conversion source/target identities
            _ - `batch_idx`
        """

        unit_series_padded, len_unit_series, mspc_series_padded, len_mspc_series, spk_embs, vc_ids = batch

        # Non teacher-forcing inference
        mspc_series_padded_predicted, len_mspc_series_predicted = self.network(unit_series_padded, len_unit_series, spk_embs)

        # Masked/normalized L1 loss
        loss = self.objective(
            mspc_series_padded_predicted,
            mspc_series_padded,
            len_mspc_series_predicted,
            len_mspc_series,
            self.device
        )
        self.log("val_loss", loss) #type: ignore ; because of PyTorch-Lightning

        # Vocoding
        mel_padded_pred_batch = mspc_series_padded_predicted.to("cpu").numpy()
        len_mel_pred_batch = len_mspc_series_predicted.to("cpu").tolist()
        for i, (mel_padded_pred, len_mel_pred) in enumerate(zip(mel_padded_pred_batch, len_mel_pred_batch)):
            mel_pred = mel_padded_pred[:len_mel_pred]
            wave_o, sr_o = self._vocoder.decode(mel_spec=mel_pred, exec_spec_norm=True)
            # [PyTorch](https://pytorch.org/docs/stable/tensorboard.html#torch.
            #     utils.tensorboard.writer.SummaryWriter.add_audio)
            self.logger.experiment.add_audio( # type: ignore ; because of PyTorch Lightning
                # e.g. `A2M_jvs001_to_jvs099_uttr00123`
                f"{vc_ids[i][3]}_{vc_ids[i][1]}_to_{vc_ids[i][0]}_{vc_ids[i][2]}",
                from_numpy(wave_o).unsqueeze(0), # snd_tensor: Tensor(1, L)
                global_step=self.global_step,
                sample_rate=sr_o,
            )

        # return anything_for_`validation_epoch_end`

    # def test_step(self, batch, batch_idx: int): # pyright: ignore [reportIncompatibleMethodOverride] ; pylint: disable=arguments-differ
    #     """(PL API) Test a batch. If not provided, test_step == validation_step."""
    #     return anything_for_`test_epoch_end`

    def predict_step(self, batch: Tuple[Tensor, Tensor]) -> Tensor: # pyright: ignore [reportIncompatibleMethodOverride] ; pylint: disable=arguments-differ
        """(PL API) Generate a mel-spectrogram from a unit sequence and speaker embedding.
        Args:
            batch
                unit_series::Tensor[Batch==1, TimeUnit, Feat] - Input unit sequence
                target_emb::Tensor[Batch==1, Emb] - Target style embedding
        Returns:
            Tensor[Batch==1, TimeMel, Freq] - mel-spectrogram
        """
        unit_series, target_emb = batch
        return self.network(unit_series.to(self.device), target_emb.to(self.device))

    def configure_optimizers(self): # type: ignore ; because of PyTorch-Lightning (no return typing, so inferred as Void)
        """Set up a optimizer
        """

        optim = AdamW(self.network.parameters(), lr=self._conf.optim.learning_rate)

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
        Convert Taco2AR-compatible mel-spectrogram to RNNMS-compatible one.

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

    def wavs2emb(self, waves: List[NDArray[np.float32]]) -> Tensor:
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
        processed_waves = [preprocess_wav(wave) for wave in waves] #type: ignore
        ave_emb: NDArray[np.float32] = self.uttr_encoder.embed_speaker(processed_waves) #type: ignore

        return from_numpy(ave_emb).unsqueeze(dim=0)
