"""Dataset"""


import random
from typing import Any, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import pickle
from hashlib import md5

import librosa # pyright: ignore [reportMissingTypeStubs]; bacause of librosa
import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler # pyright: ignore [reportMissingTypeStubs]; bacause of sklearn
from tqdm import tqdm # pyright: ignore [reportMissingTypeStubs]; bacause of tqdm
from speechdatasety.helper.archive import try_to_acquire_archive_contents, save_archive # pyright: ignore [reportMissingTypeStubs]; bacause of speechdatasety
from speechdatasety.helper.adress import dataset_adress, generate_path_getter # pyright: ignore [reportMissingTypeStubs]; bacause of speechdatasety
from speechdatasety.interface.speechcorpusy import ItemId # pyright: ignore [reportMissingTypeStubs]; bacause of speechdatasety
from omegaconf import MISSING, SI
import torch
from torch import device, tensor, from_numpy, stack # pyright: ignore [reportUnknownVariableType] ; because of PyTorch; pylint: disable=no-name-in-module
import torch.cuda
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
from resemblyzer import preprocess_wav, VoiceEncoder # pyright: ignore [reportMissingTypeStubs, reportUnknownVariableType]; bacause of resemblyzer
import s3prl.hub as hub

from .preprocessing import logmelspectrogram, ConfMelspec
from .pairs import source_to_self_target, all_source_no1_to_all_targets # pyright: ignore [reportMissingTypeStubs]
from .corpora import CorpusData


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

@dataclass
class Stat:
    """Spectrogarm statistics container"""
    mean_: NDArray[np.float32]
    scale_: NDArray[np.float32]


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
VcIdentity = Tuple[str, str, str, str]

# [unit series / mel-spectrogram / speaker embedding / voice conversion identity]
UnitMelEmbVc = Tuple[UnitSeries, Spec, SpkEmb, VcIdentity]
##########################################################################################

@dataclass
class ConfUnitMelEmbVcDataset:
    """
    Args:
        adress_data_root: Root adress of datasets
        len_chunk: Length of datum chunk, no-chunking when None
        n_shift
        sr_for_unit - sampling rate of waveform for unit
        sr_for_mel - sampling rate of waveform for mel-spectrogram
        mel - Configuration of mel-spectrogram
    """
    adress_data_root: str = MISSING
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

    def __init__(self, split: str, conf: ConfUnitMelEmbVcDataset, corpuses: Tuple[CorpusData, CorpusData]):
        """
        Prepare voice conversion tuples (source-targets tuple), then generate speaker embedding.

        Args:
            split - "train" | "dev" | "test"
            conf - Configuration
            corpus - Corpus
        """
        super().__init__()

        self.split = split
        self._len_chunk = conf.len_chunk
        self._n_shift = conf.n_shift
        self._sr_for_unit = conf.sr_for_unit
        self._sr_for_mel = conf.sr_for_mel
        self.conf_mel = conf.mel

        # Corpus & Utterances
        corpus_data_seen, corpus_data_unseen = corpuses
        self._corpus_seen = corpus_data_seen.corpus
        self._uttrs_seen = corpus_data_seen.utterances
        self._corpus_unseen = corpus_data_unseen.corpus
        self._uttrs_unseen = corpus_data_unseen.utterances

        # Hack for utterance path access
        def _get_item_path(item_id: ItemId, flag: str) -> Path:
            """A: any(unseen), M: many(seen), O: one(seen)"""
            _corpus = corpus_data_unseen.corpus if (flag is "A") else corpus_data_seen.corpus
            return _corpus.get_item_path(item_id)
        self._get_item_path = _get_item_path

        preprocessing_setup = f"{conf.n_shift}{conf.sr_for_unit}{conf.sr_for_mel}{conf.mel}"
        corpus_split_setup = f"{split}{str(self._uttrs_seen)}{str(self._uttrs_unseen)}"
        exp_specifier = md5((preprocessing_setup+corpus_split_setup).encode()).hexdigest()

        # Construct dataset adresses
        adress_archive, self._path_contents = dataset_adress(
            conf.adress_data_root,
            f"{self._corpus_seen.__class__.__name__}_{self._corpus_unseen.__class__.__name__}",
            "unit_mel_emb",
            exp_specifier,
        )
        self._get_path_unit = generate_path_getter("unit", self._path_contents)
        self._get_path_emb = generate_path_getter("emb", self._path_contents)
        self._get_path_mel = generate_path_getter("mel", self._path_contents)
        self._path_stats = self._path_contents / "stats.pkl"

        # vc_tuples: list of [content_source_utt, style_target_utt_1, style_target_utt_2, ...]
        if split == "train":
            o2o_pairs = source_to_self_target(self._uttrs_seen)
            self._vc_pairs = o2o_pairs
        else: # val/test
            a2m_pairs = all_source_no1_to_all_targets(self._uttrs_unseen, self._uttrs_seen,   ("A","M"))
            a2a_pairs = all_source_no1_to_all_targets(self._uttrs_unseen, self._uttrs_unseen, ("A","A"))
            self._vc_pairs = a2m_pairs + a2a_pairs
        # Preprocessing subjects (unique)
        ## sources: content source, which will be preprocessed as unit series
        self._sources: List[Tuple[ItemId, str]] = list(set(map(lambda vc_pair: (vc_pair.source, vc_pair.setup[0]), self._vc_pairs)))
        ## targets: style target, which will be preprocessed as embedding
        self._targets: List[Tuple[ItemId, str]] = []
        for pair in self._vc_pairs:
            self._targets += list(map(lambda target: (target, pair.setup[1]), pair.targets))
        self._targets = list(set(self._targets))

        # Deploy dataset contents.
        contents_acquired = try_to_acquire_archive_contents(adress_archive, self._path_contents)
        if not contents_acquired:
            # Generate the dataset contents from corpus
            print("Dataset archive file is not found. Automatically generating new dataset...")
            self._generate_dataset_contents()
            save_archive(self._path_contents, adress_archive)
            print("Dataset contents was generated and archive was saved.")

    def _generate_dataset_contents(self) -> None:
        """Generate dataset with corpus auto-download and preprocessing.
        """

        self._corpus_seen.get_contents()
        self._corpus_unseen.get_contents()

        # Unit
        _device = device("cuda") if torch.cuda.is_available() else device("cpu")
        model_unit = getattr(hub, 'vq_wav2vec')().to(_device)
        for item_id, setup in tqdm(self._sources, desc="Preprocess/unit", unit="utterance"): # type: ignore ; because of tqdm
            item_id: ItemId = item_id # For typing
            setup: str = setup # For typing
            wave: NDArray[np.float32] = librosa.load(self._get_item_path(item_id, setup), sr=self._sr_for_unit)[0] # type: ignore ; because of librosa
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
        for item_id, setup in tqdm(self._targets, desc="Preprocess/Embedding", unit="utterance"): # type: ignore ; because of tqdm
            item_id: ItemId = item_id # For typing
            setup: str = setup # For typing
            wav = preprocess_wav(self._get_item_path(item_id, setup)) # type: ignore ; because of resemblyzer
            embedding = spk_encoder.embed_utterance(wav) # type: ignore ; because of resemblyzer
            write_npy(self._get_path_emb(item_id), embedding.astype(np.float32))

        # Mel-spectrogram
        for item_id, setup in tqdm(self._sources, desc="Preprocess/Melspectrogram", unit="utterance"): # type: ignore ; because of tqdm
            item_id: ItemId = item_id # For typing
            setup: str = setup # For typing
            # Resampling for mel-spec generation
            wave = librosa.load(self._get_item_path(item_id, setup), sr=self._sr_for_mel)[0] # type: ignore ; because of librosa
            # lmspc::(Time, Freq) - Mel-frequency Log(ref=1, Bel)-amplitude spectrogram
            lmspc = logmelspectrogram(wave=wave, conf=self.conf_mel)
            write_npy(self._get_path_mel(item_id), lmspc)

        # Statistics
        if self.split == "train":
            self._calculate_spec_stat()
            print("Preprocess/Stats (only in `train`) - done")

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
        statman = StandardScaler() # type: ignore ; temporarily

        # average spectrum over source utterances :: (MelFreq)
        spec_stack = None
        total_len = 0
        for item_id, _ in self._sources:
            # lmspc::(Time, MelFreq)
            lmspc = read_npy(self._get_path_mel(item_id))
            uttr_sum = np.sum(lmspc, axis=0) # type: ignore ; temporarily
            spec_stack = np.add(spec_stack, uttr_sum) if spec_stack is not None else uttr_sum
            total_len += lmspc.shape[0]
        ave = spec_stack/total_len # type: ignore ; temporarily

        ## sigma in each frequency bin :: (MelFreq)
        sigma_stack = None
        total_len = 0
        for item_id, _ in self._sources:
            # lmspc::(Time, MelFreq)
            lmspc = read_npy(self._get_path_mel(item_id))
            uttr_sigma_sum = np.sum(np.abs(lmspc - ave), axis=0) # type: ignore ; temporarily
            sigma_stack = np.add(sigma_stack, uttr_sigma_sum) if sigma_stack is not None else uttr_sigma_sum
            total_len += lmspc.shape[0]
        sigma = sigma_stack / total_len # type: ignore ; temporarily

        scaler = Stat(ave, sigma) # type: ignore ; temporarily

        # Save
        with open(self._path_stats, "wb") as f:
            pickle.dump(scaler, f)

    def __len__(self) -> int:
        """Number of .wav files (and same number of embeddings)"""
        return len(self._vc_pairs)

    def __getitem__(self, index: int) -> UnitMelEmbVc:
        """Load data::UnitMelEmbVc.

        Returns - UnitSeries/MelSpectrogram/speakerEmbedding/VCidentity
        """

        # Data id
        selected = self._vc_pairs[index]
        source_id = selected.source
        target_ids = selected.targets
        setup = selected.setup

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

        # VC identity             (target_speaker,        source_speaker,    utterance_name, setup)
        vc_identity: VcIdentity = (target_ids[0].speaker, source_id.speaker, source_id.name, f"{setup[0]}2{setup[1]}")

        return unit_series, lmspc, spk_emb, vc_identity

    def collate_fn(self, batch: List[UnitMelEmbVc]):
        """collate function used by dataloader.

        Sort data with feature time length, then pad features.
        Args:
            batch: (B, UnitSeries, lmspc::[Time, Freq], spk_emb, vc_identity)
        Returns:
            unit_series_padded :: (Batch, Tmax, Feat) - Padded unit series
            len_unit_series    :: (Batch,)            - Time lengths of each unit series
            mspc_series_padded :: (Batch, Tmax, Freq) - Padded mel-spectrograms
            len_mspc_series    :: (Batch,)            - Time lengths of each mel-spectrogram
            spk_embs           :: (Batch, Emb)        - Speaker embeddings
            vc_ids: (target_speaker, source_speaker, utterance_name)[]
        """

        # Sort for RNN PackedSequence
        sorted_batch = sorted(batch, key=lambda item: -item[0].shape[0])

        # Tensor-nization of each item
        unit_series =    [from_numpy(unitMelEmbVc[0]) for unitMelEmbVc in sorted_batch]
        mspc_series =    [from_numpy(unitMelEmbVc[1]) for unitMelEmbVc in sorted_batch]
        # Padding and batching :: (Ti, Feat)[] -> (Batch, Tmax, Feat)
        unit_series_padded = pad_sequence(unit_series, batch_first=True)
        mspc_series_padded = pad_sequence(mspc_series, batch_first=True)
        # Length of series for RNN PackedSequence
        len_unit_series = tensor([series.size(0) for series in unit_series])
        len_mspc_series = tensor([series.size(0) for series in mspc_series])

        spk_embs = stack([from_numpy(unitMelEmbVc[2]) for unitMelEmbVc in sorted_batch])
        vc_ids =                    [unitMelEmbVc[3]  for unitMelEmbVc in sorted_batch]

        return unit_series_padded, len_unit_series, mspc_series_padded, len_mspc_series, spk_embs, vc_ids
