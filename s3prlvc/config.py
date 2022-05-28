"""S3PRL-VC global configurations"""


from typing import Optional
from dataclasses import dataclass

from omegaconf import MISSING

from .config_gen import generate_conf_loader

from .train import ConfTrain
from .model import ConfTaco2ARVC
from .data.datamodule import ConfWavMelEmbVcData


CONF_DEFAULT_STR = """
sr_for_unit: 16000
dim_unit: 512
upstream_rate: 160
dim_mel: 80
sr_for_mel: 24000
mel_hop_length: 256
train_steps: 50000
model:
sr_for_unit: "${sr_for_unit}"
unit_hop_length: "${upstream_rate}"
sr_for_mel: "${sr_for_mel}"
mel_hop_length: "${mel_hop_length}"
net:
    dim_latent: 1024
    dim_processed_ar: 256
    dim_o: "${dim_mel}"
    encoder:
    dim_i: "${dim_unit}"
    causal: False
    num_conv_layers: 3
    conv_dim_c: 512
    conv_size_k: 5
    conv_batch_norm: True
    conv_residual: False
    conv_dropout_rate: 0.5
    bidirectional: True
    num_rnn_layers: 1
    global_cond:
    integration_type: concat
    dim_global_cond: 256
    dec_prenet:
    n_layers: 2
    dropout_rate: 0.5
    dec_mainnet:
    dim_h: 1024
    num_layers: 2
    dropout_rate: 0.2
    layer_norm: False
    projection: True
optim:
    learning_rate: 1.0e-4
    sched_warmup_step: 4000
    sched_total_step: "${train_steps}"
data:
adress_data_root: "/content/gdrive/MyDrive/ML_data"
corpus:
    download: False
    train:
    name: VCTK
    val:
    name: VCTK
    test:
    name: VCC20
loader:
    batch_size_train: 6
    batch_size_val: 5
    batch_size_test: 5
    num_workers: null
    pin_memory: null
dataset:
    num_target: 10
    num_dev_sample: 5
    len_chunk: null
    # clip_length_mel: null # `len_chunk` ######################################
    n_shift: "${mel_hop_length}"
    sr_for_unit: "${sr_for_unit}"
    sr_for_mel: "${sr_for_mel}"
    mel:
    n_fft: 1024
    ref_db: 0.0
    min_db_rel: -200.0
    n_mels: "${dim_mel}"
    fmin: 80
    fmax: 7600
train:
max_epochs: 2000
# max_steps: "${train_steps}"
val_interval_epoch: 20
# val_interval_step: 10000
profiler: null
ckpt_log:
    dir_root: S3PRL_VC
    name_exp: a2a
    name_version: default
# eval_dataloaders:
#   - dev
#   - test
"""


@dataclass
class ConfGlobal:
    """Configuration of everything.
    Args:
        sr_for_unit - Sampling rate of waveform for unit generation
        dim_unit - Dimension size of unit
        upstream_rate - Rate of upstream output [unit/sec]
        dim_mel - Feature dimension size of mel-spectrogram
        sr_for_mel - Sampling rate of waveform for mel-spectrogram
        mel_hop_length - STFT hop length of mel-spectrogram
        train_steps - The number of training steps
    """
    sr_for_unit: int = MISSING
    dim_unit: int = MISSING
    upstream_rate: int = MISSING
    dim_mel: int = MISSING
    sr_for_mel: int = MISSING
    mel_hop_length: int = MISSING
    train_steps: int = MISSING
    model: ConfTaco2ARVC = ConfTaco2ARVC()
    data: ConfWavMelEmbVcData = ConfWavMelEmbVcData()
    train: ConfTrain = ConfTrain()
    expdir: str = MISSING

    seed: int = MISSING
    path_extend_conf: Optional[str] = MISSING


# Exported
load_conf = generate_conf_loader(CONF_DEFAULT_STR, ConfGlobal)
"""Load configuration type-safely.
"""
