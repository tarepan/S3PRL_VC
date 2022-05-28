"""Dataset"""


import random
from typing import Any, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import pickle

import librosa # pyright: ignore [reportMissingTypeStubs]; bacause of librosa
import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler # pyright: ignore [reportMissingTypeStubs]; bacause of sklearn
from tqdm import tqdm # pyright: ignore [reportMissingTypeStubs]; bacause of tqdm
from speechdatasety.helper.archive import try_to_acquire_archive_contents, save_archive # pyright: ignore [reportMissingTypeStubs]; bacause of speechdatasety
from speechdatasety.helper.adress import dataset_adress, generate_path_getter # pyright: ignore [reportMissingTypeStubs]; bacause of speechdatasety
from speechdatasety.interface.speechcorpusy import AbstractCorpus, ItemId # pyright: ignore [reportMissingTypeStubs]; bacause of speechdatasety
from omegaconf import MISSING, SI
import torch
from torch import Tensor, device, from_numpy # pyright: ignore [reportUnknownVariableType] ; because of PyTorch; pylint: disable=no-name-in-module
import torch.cuda
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
from resemblyzer import preprocess_wav, VoiceEncoder # pyright: ignore [reportMissingTypeStubs, reportUnknownVariableType]; bacause of resemblyzer
import s3prl.hub as hub # pyright: ignore [reportMissingTypeStubs]

from .preprocessing import logmelspectrogram, ConfMelspec


#####################################################
# Utils

def read_npy(path: Path) -> Any:
    """Read .npy from path without `.npy`"""
    # Change file name by appending `.npy` at tail.
    return np.load(path.with_name(f"{path.name}.npy")) # type: ignore

def write_npy(path: Path, data: Any) -> None:
    """Write .npy from path"""
    path.parent.mkdir(exist_ok=True, parents=True)
    np.save(path, data) # type: ignore
#####################################################


def speaker_split_jvs(utterances: List[ItemId]) -> Tuple[List[ItemId], List[ItemId]]:
    """Split JVS corpus items into two groups."""

    anothers_spk = ["95", "96", "98", "99"]
    # Filter for train/test split of single corpus
    ones = list(filter(
        lambda item_id: item_id.speaker not in anothers_spk,
        utterances
    ))
    anothers = list(filter(
        lambda item_id: item_id.speaker in anothers_spk,
        utterances
    ))
    return ones, anothers


@dataclass
class Stat:
    """Spectrogarm statistics container"""
    mean_: NDArray[np.float32]
    scale_: NDArray[np.float32]


def save_vc_tuples(content_path: Path, num_target: int, tuples: List[List[ItemId]]):
    """Save VCTuples"""
    path = content_path / f"vc_{num_target}_tuples.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(tuples, file)

def load_vc_tuples(content_path: Path, num_target: int) -> List[List[ItemId]]:
    """Load VCTuples"""
    path = content_path / f"vc_{num_target}_tuples.pkl"
    if path.exists():
        with open(path, "rb") as file:
            vc_tuples: List[List[ItemId]] = pickle.load(file)
            return vc_tuples
    else:
        raise Exception(f"{str(path)} does not exist.")


def generate_vc_tuples(
    sources: List[ItemId],
    targets: List[ItemId],
    num_target: int,
    ) -> List[List[ItemId]]:
    """Generate utterance tuples for voice convertion.

    VC needs a source utterance (content source) and multiple target utterances (style target).
    This function generate this tuples for all source utterances.
    Target utterances are randomly sampled from utterances of single target speaker.
    Args:
        sources: Source utterances
        targets: Target utterances
        num_target: Target utterances (that will be averaged in downstream) per single source
    Returns:
        (List[ItemId]): #0 is a source, #1~ are targets
    """
    target_speakers = set(map(lambda item_id: item_id.speaker, targets))
    full_list: List[List[ItemId]] = []
    for trgspk in target_speakers:
        # "all_sources to trgspk" vc tuple
        utts_of_trgspk = list(filter(lambda item_id: item_id.speaker == trgspk, targets))
        for src in sources:
            vc_tuple = [src, *random.sample(utts_of_trgspk, num_target)]
            full_list.append(vc_tuple)
    return full_list


def random_clip_index(full_length: int, clip_length: int) -> Tuple[int, int]:
    """Calculate indexes for array clipping. Tested."""
    start = random.randrange(0, full_length - (clip_length - 1))
    end = start + clip_length
    return start, end


########################################## Data ##########################################
# Unit series
UnitSeries = NDArray[np.float32]
# Spectrogram
Spec = NDArray[np.float32]
# Speaker embedding
SpkEmb = NDArray[np.float32]
# VoiceConversion source/target identities
VcIdentity = Tuple[str, str, str]

# [unit series / mel-spectrogram / speaker embedding / voice conversion identity]
UnitMelEmbVc = Tuple[UnitSeries, Spec, SpkEmb, VcIdentity]
##########################################################################################


@dataclass
class ConfUnitMelEmbVcDataset:
    """
    Args:
        adress_data_root: Root adress of datasets
        num_target: Number of target utterances per single source utterance
        num_dev_sample: Number of dev samples per single speaker
        len_chunk: Length of datum chunk, no-chunking when None
        n_shift
        sr_for_unit - sampling rate of waveform for unit
        sr_for_mel - sampling rate of waveform for mel-spectrogram
        mel - Configuration of mel-spectrogram
    """
    adress_data_root: str = MISSING
    num_target: int = MISSING
    num_dev_sample: int = MISSING
    len_chunk: Optional[int] = None
    n_shift: int = MISSING
    sr_for_unit: int = MISSING
    sr_for_mel: int = MISSING
    mel: ConfMelspec = ConfMelspec(
        sampling_rate=SI("${..sr_for_mel}"),
        hop_length=SI("${..n_shift}"),)

class UnitMelEmbVcDataset(Dataset[UnitMelEmbVc]):
    """Dataset containing [unit series / mel-spectrogram / speaker embedding / voice conversion identity].
    """

    def __init__(self, split: str, conf: ConfUnitMelEmbVcDataset, corpus: AbstractCorpus):
        """
        Prepare voice conversion tuples (source-targets tuple), then generate speaker embedding.

        Data split: [train, dev] = [:-11, -5:, -10:-5] for each speaker

        Args:
            split - "train" | "dev" | "test"
            conf - Configuration
            corpus - Corpus
        """
        super().__init__()
        self.split = split
        self._num_target = conf.num_target
        self._len_chunk = conf.len_chunk
        self._n_shift = conf.n_shift
        self._sr_for_unit = conf.sr_for_unit
        self._sr_for_mel = conf.sr_for_mel
        self.conf_mel = conf.mel

        self._corpus = corpus
        corpus_name = self._corpus.__class__.__name__

        # Construct dataset adresses
        adress_archive, self._path_contents = dataset_adress(
            conf.adress_data_root,
            self._corpus.__class__.__name__,
            "unit_mel_emb_vctuple",
            f"{split}_{conf.num_dev_sample}forDev_{conf.num_target}targets",
        )
        self._get_path_unit = generate_path_getter("unit", self._path_contents)
        self._get_path_emb = generate_path_getter("emb", self._path_contents)
        self._get_path_mel = generate_path_getter("mel", self._path_contents)
        self._path_stats = self._path_contents / "stats.pkl"

        # Select data identities.
        #   Train: reconstruction (source_uttr + source_emb  -> source_uttr)
        #   Dev:   unseen-uttr    (source_uttr + source_emb  -> source_uttr)
        #   Test:  A2A VC         (source_uttr + ave_tgt_emb -> target_uttr)

        all_utterances = self._corpus.get_identities()
        ## list of [content_source_utt, style_target_utt_1, style_target_utt_2, ...]
        self._vc_tuples: List[List[ItemId]] = []
        ## Utterance list of content source, which will be preprocessed as resampled waveform
        self._sources: List[ItemId] = []
        ## Utterance list of style target, which will be preprocessed as embedding
        self._targets: List[ItemId] = []

        # Speaker split (train spk, val spk, test spk) & utterance split (train, val, test)
        #
        # [current] Experiment design: train / val O2O / test A2A
        #          utterance    speaker
        #   train      -           -
        #   val     unseen        M2M
        #   test    unseen        A2A
        #
        # [ideal] Experiment design: train / val A2M & A2A / test A2M & A2A
        #               spk_1s         spk_2s        spk_3s
        #   uttr_1s     <train>           -             -
        #   uttr_2s  [val M2 & 2M]  [val A2 & 2A]       -
        #              (test 2M)
        #   uttr_3s        -              -       (Test A2 & 2A)
        #
        # how to implement
        #   train: self-target reconstruction with uttr_1s/spk_1s corpus
        #   val:
        #     O2O: unseen utterance self-target reconstruction with uttr_2s/spk_1s corpus
        #     A2M: seen-target VC of unseen utterances (uttr_2s/spk_2s => uttr_2s/spk_1s)
        #     A2A: unseen-target VC of unseen utterances (uttr_2s/spk_2s => other spk of uttr_2s/spk_2s) # pylint: disable=line-too-long
        #   test:
        #     A2M: seen-target VC of train-val-unseen utterances (uttr_3s/spk_3s => uutr_2s/spk_1s (uttr_2s for only target embedding) # pylint: disable=line-too-long
        #     A2A: unseen-target VC of train-val-unseen utterances (uttr_3s/spk_3s => other spk of uttr_3s/spk_3s)


        #                           JVS1-92       JVS93-96      JVS97-100
        #                           spk_1s         spk_2s        spk_3s
        #     1-92      uttr_1s      <train>           -             -
        #    93-96      uttr_2s     [val 2M]    [val A2 & 2A]       -
        #                          (test 2M)
        #    97-100     uttr_3s        -              -       (Test A2 & 2A)
        #
        #    (max)
        #    val/test A2M: 4spk/4uttr x 92spk = 1472 wav
        #    val/test A2A: 4spk/4uttr x 4spk = 64 wav
        #
        #    val  -> often, so number should be limited ([3spk]-to-<2spk> A2M (6 uttr), [3spk]-to-[3spk] A2A (9 uttr))
        #
        #    test -> rarely recorded

        # Prepare `_sources` and `_targets`
        if split == 'train' or split == 'dev':
            # Additionally, prepare self-target `_vc_tuples`
            if corpus_name == "JVS":
                # train & val share speakers (seen speaker (O2O | M2M))
                all_utterances = speaker_split_jvs(all_utterances)[0]

            is_train = split == 'train'
            idx_dev = -1*conf.num_dev_sample
            for spk in set(map(lambda item_id: item_id.speaker, all_utterances)):
                utts_spk = filter(lambda item_id: item_id.speaker == spk, all_utterances)
                # tuples_spk = [[X#1, X#1], [X#2, X#2], ..., [X#n, X#n]]
                tuples_spk = list(map(lambda item_id: [item_id, item_id], utts_spk))
                # Data filtering 3/n
                ## Data split: [0, -2X] is for train, [-X:] is for dev for each speaker
                self._vc_tuples.extend(tuples_spk[:2*idx_dev] if is_train else tuples_spk[idx_dev:])
            # source == target
            self._sources = list(map(lambda vc_tuple: vc_tuple[0], self._vc_tuples))
            self._targets = list(map(lambda vc_tuple: vc_tuple[1], self._vc_tuples))
        elif split == 'test':
            # `_vc_tuples` is not defined here, but defined in preprocessing
            if corpus_name == "VCC20":
                # Missing utterances in original code: E10001-E10050 (c.f. tarepan/s3prl#2)
                self._sources = list(filter(lambda item_id: item_id.subtype == "eval_source", all_utterances))
                self._targets = list(filter(lambda i: i.subtype == "train_target_task1", all_utterances))
            elif corpus_name == "JVS":
                all_utterances = speaker_split_jvs(all_utterances)[1]
                # 10 utterances per speaker for test source
                self._sources = []
                for spk in set(map(lambda item_id: item_id.speaker, all_utterances)):
                    utts_spk = list(filter(lambda item_id: item_id.speaker == spk, all_utterances))
                    self._sources.extend(utts_spk[:10])
                # All test utterances are target style
                self._targets = all_utterances
            elif corpus_name == "AdHoc":
                self._sources = list(filter(lambda item_id: item_id.subtype == "s", all_utterances))
                self._targets = list(filter(lambda item_id: item_id.subtype == "t", all_utterances))
            else:
                Exception(f"Corpus '{corpus_name}' is not yet supported for test split")

        # Deploy dataset contents.
        contents_acquired = try_to_acquire_archive_contents(adress_archive, self._path_contents)
        if not contents_acquired:
            # Generate the dataset contents from corpus
            print("Dataset archive file is not found. Automatically generating new dataset...")
            self._generate_dataset_contents()
            save_archive(self._path_contents, adress_archive)
            print("Dataset contents was generated and archive was saved.")

        # Load vc tuples in the file
        if split == 'test':
            self._vc_tuples = load_vc_tuples(self._path_contents, conf.num_target)

        # Report
        print(f"[Dataset] - number of data for {split}: {len(self._vc_tuples)}")

    def _generate_dataset_contents(self) -> None:
        """Generate dataset with corpus auto-download and preprocessing.
        """

        self._corpus.get_contents()

        # Unit
        _device = device("cuda") if torch.cuda.is_available() else device("cpu")
        model_unit = getattr(hub, 'vq_wav2vec')().to(_device)
        for item_id in tqdm(self._sources, desc="Preprocess/unit", unit="utterance"): # type: ignore ; because of tqdm
            item_id: ItemId = item_id # For typing
            wave: NDArray[np.float32] = librosa.load(self._corpus.get_item_path(item_id), sr=self._sr_for_unit)[0] # type: ignore ; because of librosa
            ## wav2unit
            with torch.no_grad():
                # vq-wav2vec do not pad. Manual padding in both side is needed.
                # todo: fix hardcoding
                n_receptive_field, stride_unit = 480, 160
                pad_oneside = (n_receptive_field//stride_unit -1)//2 * stride_unit
                wave = np.pad(wave, pad_oneside, mode="reflect") # pyright: ignore [reportUnknownMemberType] ; because of numpy
                # [(pad+T_wave+pad,)] => (B=1, T_unit=T_wave//stride, vq_dim=512) => (T_unit, vq_dim=512)
                unit_series = model_unit([from_numpy(wave).to(_device)])["codewords"][0]
            write_npy(self._get_path_unit(item_id), unit_series.cpu().numpy())

        # Embedding
        spk_encoder = VoiceEncoder()
        for item_id in tqdm(self._targets, desc="Preprocess/Embedding", unit="utterance"): # type: ignore ; because of tqdm
            item_id: ItemId = item_id # For typing
            wav = preprocess_wav(self._corpus.get_item_path(item_id)) # type: ignore ; because of resemblyzer
            embedding = spk_encoder.embed_utterance(wav) # type: ignore ; because of resemblyzer
            write_npy(self._get_path_emb(item_id), embedding.astype(np.float32))

        # Mel-spectrogram
        for item_id in tqdm(self._sources, desc="Preprocess/Melspectrogram", unit="utterance"): # type: ignore ; because of tqdm
            item_id: ItemId = item_id # For typing
            # Resampling for mel-spec generation
            wave = librosa.load(self._corpus.get_item_path(item_id), sr=self._sr_for_mel)[0] # type: ignore ; because of librosa
            # lmspc::(Time, Freq) - Mel-frequency Log(ref=1, Bel)-amplitude spectrogram
            lmspc = logmelspectrogram(wave=wave, conf=self.conf_mel)
            write_npy(self._get_path_mel(item_id), lmspc)

        # Statistics
        if self.split == "train":
            self._calculate_spec_stat()
            print("Preprocess/Stats (only in `train`) - done")

        # VC tuples
        if self.split == "test":
            # Generate vc tuples randomly (source:target = 1:num_target)
            vc_tuples = generate_vc_tuples(self._sources, self._targets, self._num_target)
            save_vc_tuples(self._path_contents, self._num_target, vc_tuples)
            print("Preprocess/VC_tuple (only in `test`) - done")

    def acquire_spec_stat(self):
        """Acquire scaler, the statistics (mean and variance) of mel-spectrograms"""
        with open(self._path_stats, "rb") as f:
            scaler =  pickle.load(f)
        return scaler

    def _calculate_spec_stat(self):
        """Calculate mean and variance of source non-standardized spectrograms."""

        # Implementation Notes:
        #   Dataset could be huge, so loading all spec could cause memory overflow.
        #   For this reason, this implementation repeat 'load a spec and stack stats'.
        #   TODO: Fix wrong SD calculation (now: Sum of abs, correct: Root of square sum)
        #   TODO: replace with `StandardScaler().partial_fit` (for bug fix and simplification)

        # For future fix
        statman = StandardScaler()

        # average spectrum over source utterances :: (MelFreq)
        spec_stack = None
        L = 0
        for item_id in self._sources:
            # lmspc::(Time, MelFreq)
            lmspc = read_npy(self._get_path_mel(item_id))
            uttr_sum = np.sum(lmspc, axis=0)
            spec_stack = np.add(spec_stack, uttr_sum) if spec_stack is not None else uttr_sum
            L += lmspc.shape[0]
        ave = spec_stack/L

        ## sigma in each frequency bin :: (MelFreq)
        sigma_stack = None
        L = 0
        for item_id in self._sources:
            # lmspc::(Time, MelFreq)
            lmspc = read_npy(self._get_path_mel(item_id))
            uttr_sigma_sum = np.sum(np.abs(lmspc - ave), axis=0)
            sigma_stack = np.add(sigma_stack, uttr_sigma_sum) if sigma_stack is not None else uttr_sigma_sum
            L += lmspc.shape[0]
        sigma = sigma_stack/L

        scaler = Stat(ave, sigma)

        # Save
        with open(self._path_stats, "wb") as f:
            pickle.dump(scaler, f)

    def __len__(self) -> int:
        """Number of .wav files (and same number of embeddings)"""
        return len(self._vc_tuples)

    def __getitem__(self, index: int) -> UnitMelEmbVc:
        """Load data::UnitMelEmbVc.

        Returns - UnitSeries/MelSpectrogram/speakerEmbedding/VCidentity
        """

        # Data id
        selected = self._vc_tuples[index]
        source_id = selected[0]
        target_ids = selected[1:]

        # Unit series & Mel-spec
        unit_series: UnitSeries = read_npy(self._get_path_unit(source_id))
        lmspc: Spec = read_npy(self._get_path_mel(source_id))
        ## Chunked training
        if (self.split == "train") and (self._len_chunk is not None):
            # Time-directional random clipping ::(T_mel, freq) -> (clip_mel, freq)
            start_mel, end_mel = random_clip_index(lmspc.shape[-2], self._len_chunk)
            lmspc = lmspc[start_mel : end_mel]

            # Waveform clipping
            wav_length = len(input_wav_resample)
            effective_stride = self._n_shift * (self._sr_for_unit / self._sr_for_mel)
            start_wave = max(0, round(effective_stride * start_mel))
            end_wave = min(wav_length, round(effective_stride * end_mel) + 1)
            input_wav_resample = input_wav_resample[start_wave : end_wave]

        # Spaeker embedding: averaged embedding of the speaker X's utterances (N==`len(target_ids)`)
        spk_embs: List[SpkEmb] = [read_npy(self._get_path_emb(item_id)) for item_id in target_ids]
        spk_emb: SpkEmb = np.mean(np.stack(spk_embs, axis=0), axis=0) # type:ignore

        # VC identity             (target_speaker,        source_speaker,    utterance_name)
        vc_identity: VcIdentity = (target_ids[0].speaker, source_id.speaker, source_id.name)

        return unit_series, lmspc, spk_emb, vc_identity

    def collate_fn(self, batch: List[UnitMelEmbVc]):
        """collate function used by dataloader.

        Sort data with feature time length, then pad features.
        Args:
            batch: (B, UnitSeries, lmspc::[Time, Freq], spk_emb, vc_identity)
        Returns:
            wavs: List[Tensor(`input_wav_resample`)]
            acoustic_features: List[lmspc::Tensor[Time, Freq]]
            acoustic_features_padded: `acoustic_features` padded by PyTorch function
            acoustic_feature_lengths: Tensor[Batch,] - signal length of acoustic_features
            spk_embs: Tensor(`spk_emb`)
            vc_ids: List[(target_speaker, source_speaker, utterance_name)]
        """

        # Sort for RNN packing
        sorted_batch = sorted(batch, key=lambda item: -item[0].shape[0])

        # # Padding
        # # input_feature_lengths::(B, T)
        # input_feature_lengths = torch.IntTensor([feature.shape[0] for feature in unit_series])
        # # (T, Feat)[] -> (B, Tmax, Feat)
        # input_features = pad_sequence(input_features, batch_first=True)

        units =                 list(map(lambda item: from_numpy(item[0]), sorted_batch))
        acoustic_features =     list(map(lambda item: from_numpy(item[1]), sorted_batch))

        unit_series_padded = pad_sequence(units, batch_first=True)
        unit_lengths = from_numpy(np.array(list(map(lambda series: series.size(0), units)), dtype=np.int64))
        acoustic_features_padded = pad_sequence(acoustic_features, batch_first=True)
        acoustic_feature_lengths = from_numpy(np.array(list(map(lambda feat: feat.size(0), acoustic_features))))
        spk_embs =     from_numpy(np.array(list(map(lambda item: item[2],  sorted_batch))))
        vc_ids =         list(map(lambda item:                   item[3],  sorted_batch))

        return unit_series_padded, unit_lengths, acoustic_features_padded, acoustic_feature_lengths, spk_embs, vc_ids
