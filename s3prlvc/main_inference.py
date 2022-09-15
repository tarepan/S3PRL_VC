"""Run S3PRL-VC inference"""


import argparse
import torch
from torch import inference_mode
import torchaudio
import librosa # pyright: reportMissingTypeStubs=false
import soundfile as sf

from .model import Taco2ARVC


if __name__ == "__main__":  # pragma: no cover

    parser = argparse.ArgumentParser(description='Run RNNMS inference')
    parser.add_argument("-m", "--model-ckpt-path",     required=True)
    parser.add_argument("-ic", "--i-wav-content-path", required=True)
    parser.add_argument("-is", "--i-wav-style-path",   required=True)
    parser.add_argument("-o", "--o-wav-path", default="reconstructed.wav")
    args = parser.parse_args()

    torchaudio.set_audio_backend("sox_io")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = Taco2ARVC.load_from_checkpoint(checkpoint_path=args.model_ckpt_path).to(device) # type: ignore ; because of PyTorch Lightning
    wave_style, style_org_sr = librosa.load(args.i_wav_style_path, sr=None)

    with inference_mode():
        unit_series = model.wav2unit(args.i_wav_content_path)
        spk_emb = model.wavs2emb([wave_style], style_org_sr)
        o_wave, o_sr = model.predict_step((unit_series, spk_emb))

    sf.write(args.o_wav_path, o_wave, o_sr)
