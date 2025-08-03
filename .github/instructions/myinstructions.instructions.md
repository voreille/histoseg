---
applyTo: '**'
---

# INSTRUCTIONS.md

## 📛 Project Name
**histoseg** – A modular and extensible codebase for semantic segmentation in histopathology using Mask2Former and ViT backbones like UNI2.

---

## 🎯 Goal
Build a PyTorch Lightning-based research codebase for semantic segmentation using:
- **Mask2Former architecture**, adapted for newer versions of PyTorch and `torch.compile`
- **UNI2-h or other ViT-based backbones** from HuggingFace/TIMM
- **Custom ViT Adapter module** and **Pixel Decoder**
- Multi-resolution feature maps passed to **MSDeformAttn** layers
- Configurable via **Lightning CLI** + YAML files
- Easily extensible to new datasets (ADE20K, custom histo data, etc.)

---

## 🧱 Repository Structure

```
histoseg/
├── histoseg/                  # Main Python module
│   ├── __init__.py
│   ├── models/
│   │   ├── encoder/           # UNI2, ViTAdapter
│   │   ├── decoder/           # Mask2Former, pixel decoder
│   │   ├── ops/               # MSDeformAttn custom CUDA
│   ├── data/
│   │   ├── dm_ade20k.py       # LightningDataModule for ADE20K
│   │   ├── transforms.py
│   ├── training/
│   │   ├── mask2former_module.py # LightningModule
│   │   ├── losses.py
│   │   └── metrics.py
│   └── utils/
│       └── io.py              # load/save checkpoints, config logging
├── configs/
│   ├── ade20k.yaml
│   ├── vit_adapter.yaml
│   └── trainer.yaml
├── cli.py                     # Main Lightning CLI launcher
├── README.md
├── setup.py
├── requirements.txt
└── compile_ops.sh            # script to build MSDeformAttn
```

---

## 🔧 Components to Implement

### 1. ✅ ViT Adapter Backbone
Use the ViT Adapter code (from InternImage or MMDet) and:
- Remove unused SyncBN if needed
- Use `timm` for patch embedding
- Make the `forward()` method return multi-scale features `[c1, c2, c3, c4]`
- Integrate with HuggingFace model loading (`timm.create_model("hf-hub:MahmoodLab/UNI2-h")`)

---

### 2. ✅ Mask2Former Decoder
- Takes multi-scale features from ViTAdapter
- Pixel decoder implementation (reuse or adapt `Mask2FormerPixelDecoder`)
- Transformer decoder with learned queries
- Mask prediction and class logits

---

### 3. ✅ MSDeformAttn Module
- Compile the CUDA ops as part of the repo
- Create `compile_ops.sh` to compile ops automatically
- Use `torch.compile` compatibility (`set_requires_grad` or other patches as needed)

---

### 4. ✅ Lightning Integration
Use PyTorch Lightning CLI:
- `cli.py` launches training
- `Mask2formerModule(LightningModule)` handles loss, forward, optimizer, and scheduler
- Use `pl.LightningDataModule` for datasets
- Support Weights & Biases logging and checkpoints

---

### 5. ✅ Configurable YAML Setup
Each experiment is launched using:

```bash
python cli.py fit --config configs/ade20k.yaml
```

Example YAML:

```yaml
trainer:
  max_steps: 40000
  val_check_interval: 1000

model:
  class_path: histoseg.training.mask2former_module.Mask2formerModule
  init_args:
    encoder:
      class_path: histoseg.models.encoder.vit_adapter.ViTAdapter
    decoder:
      class_path: histoseg.models.decoder.mask2former_decoder.Mask2formerDecoder

data:
  class_path: histoseg.data.dm_ade20k.ADE20KDataModule
```

---

## ⚠️ Constraints
- Torch >= 2.1 and should support `torch.compile`
- Compatible with `PyTorch Lightning >= 2.2`
- Use HuggingFace models via `timm.create_model("hf-hub:...")`
- Avoid legacy Detectron2 code
- Use `wandb`, `tensorboard`, or local logging
- Keep everything Pythonic and modular

---

## 🧪 Optional Goals
- Add unit tests in `tests/`
- Add support for custom histopathology datasets (e.g., LungHist700)
- Add shell script for downloading ADE20K or custom dataset

---

## 💬 CLI Usage Examples

**Training**:
```bash
python cli.py fit --config configs/ade20k.yaml
```

**Evaluating**:
```bash
python cli.py validate --ckpt_path path/to/checkpoint.ckpt
```

**Debug training**:
```bash
python cli.py fit --config configs/ade20k.yaml --fast_dev_run
```

---

## 📦 Dependencies

In `requirements.txt`:

```
torch>=2.1
pytorch-lightning>=2.2
timm
transformers
wandb
omegaconf
hydra-core
einops
```

---

## ✨ Summary

This project builds a clean, modular PyTorch Lightning implementation of Mask2Former using modern ViT backbones like UNI2 from HuggingFace. It supports YAML-based configuration, easy CLI usage, and custom CUDA ops for deformable attention. The goal is to provide a clean, hackable research framework for histopathology segmentation tasks.
