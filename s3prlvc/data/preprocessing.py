"""Preprocessings"""

from typing import Optional
from dataclasses import dataclass

import numpy as np
from omegaconf import MISSING
import librosa
import librosa.feature


def db_to_linear(decibel: float) -> float:
    """Convert level [dB(ref=1,power)] to linear"""
    return 10**(decibel/20.)


@dataclass
class ConfMelspec:
    """
    Configuration of mel-spectrogram
    Args:
        sampling_rate - waveform sampling rate
        n_fft - Length of FFT chunk
        hop_length - STFT hop length
        ref_db - Reference level [dB(ref=1,power)]
        min_db_rel - Minimum level relative to reference [dB(ref=1,power)]
        n_mels - Dimension size of mel frequency
        fmin - Minumum frequency of mel spectrogram
        fmax - Maximum frequency of mel spectrogram, None==sr/2
    """
    sampling_rate: int = MISSING
    n_fft: int = MISSING
    hop_length: int = MISSING
    ref_db: float = MISSING
    min_db_rel: float = MISSING
    n_mels: int = MISSING
    fmin: int = MISSING
    fmax: Optional[int] = MISSING

def logmelspectrogram(wave: np.ndarray, conf: ConfMelspec) -> np.ndarray:
    """Convert a waveform to a scaled mel-frequency log-amplitude spectrogram.

    Args:
        wave::ndarray[Time,] - waveform
        conf - Configuration
    Returns::(Time, Mel_freq) - mel-frequency log(Bel)-amplitude spectrogram
    """
    # mel-frequency linear-amplitude spectrogram :: [Freq=n_mels, T_mel]
    mel_freq_amp_spec = librosa.feature.melspectrogram(
        y=wave,
        sr=conf.sampling_rate,
        n_fft=conf.n_fft,
        hop_length=conf.hop_length,
        n_mels=conf.n_mels,
        fmin=conf.fmin,
        fmax=conf.fmax,
        # norm=,
        power=1,
        pad_mode="reflect",
    )
    # [-inf, `min_db`, `ref_db`, +inf] dB(ref=1,power) => [`min_db_rel`/20, `min_db_rel`/20, 0, +inf]
    min_db = conf.ref_db + conf.min_db_rel
    ref, amin = db_to_linear(conf.ref_db), db_to_linear(min_db)
    # `power_to_db` hack for linear-amplitude spec to log-amplitude spec conversion
    mel_freq_log_amp_spec = librosa.power_to_db(mel_freq_amp_spec, ref=ref, amin=amin, top_db=None)
    mel_freq_log_amp_spec_bel = mel_freq_log_amp_spec/10.
    mel_freq_log_amp_spec_bel = mel_freq_log_amp_spec_bel.T
    return mel_freq_log_amp_spec_bel
