# InsightFace iResNet

Source: https://raw.githubusercontent.com/deepinsight/insightface/d9cb10e6bda109bc9b4a6ca8b92d7f3342b56492/recognition/arcface_torch/backbones/iresnet.py

Revision: `d9cb10e6bda109bc9b4a6ca8b92d7f3342b56492`

Source SHA-256: `fb57d145c4c26d8e55680c24fa164541a7222ccdefa3a1716c054ff4c5859753`

Unmodified source, MIT license in LICENSE.insightface. Pretrained weights are separate and subject to the upstream non-commercial research terms.

## Pretrained weights

`checkpoints/arcface/iresnet50_w600k.pt` (not in git, not redistributed) is produced locally by `tools/convert_w600k_onnx.py` from the InsightFace `buffalo_l` model pack, file `w600k_r50.onnx` (ArcFace iResNet50 trained on WebFace600K, SHA-256 `4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43`). If the pack is absent, the converter downloads it from https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip and checks SHA-256 `80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f`.

InsightFace pretrained models, including `buffalo_l`, the converted checkpoint and classifiers built on it, are available for **non-commercial research purposes only**. The MIT license above covers the code only, not the weights.
