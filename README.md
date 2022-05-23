<div align="center">

# S3PRL-VC : Intra-lingual A2A VC <!-- omit in toc -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][notebook]
[![Paper](http://img.shields.io/badge/paper-arxiv.2110.06280-B31B1B.svg)][paper]

</div>

Intra-lingual Any-to-Any Voice Conversion based on S3PRL; S3PRL-VC.  
This repository is PyTorch Lightning based reimplementation of [*official S3PRL-VC*][official_s3prlvc].  

## Task
The VCC2020 Task1; Intra-lingual any-to-any voice conversion.  
Trained on VCTK, evaluated on VCC2020.  

## Implementation

- model:
  - wave2mel: any S3PRL upstreams
  - unit2mel: Taco2-AR
    - speaker: [Resemblyzer]&#8203; ([d-vector])
  - mel2wave: [HiFi-GAN], [kan-bayashi's implementation][HiFi-GAN_impl]

[d-vector]: https://static.googleusercontent.com/media/research.google.com/zh-TW//pubs/archive/41939.pdf
[Resemblyzer]: https://github.com/resemble-ai/Resemblyzer
[HiFi-GAN]: https://arxiv.org/abs/2010.05646
[HiFi-GAN_impl]: https://github.com/kan-bayashi/ParallelWaveGAN

## Quick Training
<!-- Jump to ☞ [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][notebook], then Run. That's all!   -->

## How to Use

<!-- ### 1. Install
First you should set up `s3prl` itself (follow instructions in root README), then execute the instruction below.  

```bash
cd <root-to-s3prl>/s3prl/downstream/a2a-vc-vctk
pip install -r requirements.txt
```

### 2. Data & Preprocessing
```bash
# Download the pretrained HiFi-GAN.
./vocoder_download.sh ./
```

### 3. Training
`<upstream>` is wave2unit model, `<tag>` is arbitrary name.  

```bash
cd ../..
./downstream/a2a-vc-vctk/vc_train.sh <upstream> downstream/a2a-vc-vctk/config_ar_taco2.yaml <tag>
```

### 4. Evaluation: Waveform synthesis & objective metrics

#### 4-A. Waveform synthesis and objective evaluation
Synthesize waveforms from already generated spectrograms and objectively evaluate them.  

```bash
cd <root-to-s3prl>/s3prl
./downstream/a2a-vc-vctk/decode.sh <vocoder> <result_dir>/<step>
# e.g. 
# ./downstream/a2a-vc-vctk/decode.sh ./downstream/a2a-vc-vctk/hifigan_vctk result/downstream/a2a_vc_vctk_taco2_ar_decoar2/50000
```

#### 4-B. Search best epoch
Run 4-A over epochs and report the best epoch.  
```bash
cd <root-to-s3prl>/s3prl
./downstream/a2a-vc-vctk/batch_vc_decode.sh <upstream> taco2_ar downstream/a2a-vc-vctk/hifigan_vctk
```

- output1: speech samples @ `<root-to-s3prl>/s3prl/result/downstream/a2a_vc_vctk_taco2_ar_<upstream>/<step>/hifigan_wav/`
- output2: stdout (e.g. `decoar2 10 samples epoch 48000 best: 9.28 41.80 0.197 1.3 4.0 27.00`)
- output3: detailed utterance-wise results @ `<root-to-s3prl>/s3prl/result/downstream/a2a_vc_vctk_taco2_ar_<upstream>/<step>/hifigan_wav/obj_10samples.log`

If the command fails, please make sure there are trained results in `result/downstream/a2a_vc_vctk_<tag>_<upstream>/`.
 -->
## Change from original s3prl-vc
- Waveforms for melspec are resampled with `fbank_config["fs"]` (original: `sr=None`)
  - STFT parameters depends on sampling rate, so raw waveform should have intended sr

## References
### Original paper
```
@article{huang2021s3prl,
  title={S3PRL-VC: Open-source Voice Conversion Framework with Self-supervised Speech Representations},
  author={Huang, Wen-Chin and Yang, Shu-Wen and Hayashi, Tomoki and Lee, Hung-Yi and Watanabe, Shinji and Toda, Tomoki},
  journal={arXiv preprint arXiv:2110.06280},
  year={2021}
}
```
### Acknowlegements
- [s3prl/a2a-vc-vctk][official_s3prlvc]: Model and hyperparams are totally based on this official repository.

[paper]: https://arxiv.org/abs/2110.06280
[notebook]: https://colab.research.google.com/github/tarepan/S3PRL_VC/blob/main/s3prlvc.ipynb
[official_s3prlvc]: https://github.com/s3prl/s3prl/tree/master/s3prl/downstream/a2a-vc-vctk
