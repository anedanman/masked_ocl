# Masked Object-Centric Learning

Research project exploring the intersection of **object-centric learning** (slot attention) and **masked self-supervised learning** (MAE/JEPA).

## Overview

This project combines slot attention mechanisms with masked autoencoding objectives. The key idea is to use slot-based token assignments to guide masking—instead of random masking, tokens are masked per-slot, encouraging the model to learn object-centric representations through reconstruction.

### Two Approaches

| Variant | Description |
|---------|-------------|
| **Slot-MAE** | Single model performs slot attention on full features, generates per-slot masks, then reconstructs masked tokens using the same slots |
| **Slot-JEPA** | Teacher-student setup where an EMA teacher provides slot assignments on full features, guiding a student that only sees unmasked tokens |

## Architecture

```
Image → DINOv3 (frozen) → Slot Attention → Per-Slot Masking → Decoder → Reconstruction Loss
```

- **Backbone**: Frozen DINOv3 features (ViT-S/B/L)
- **Slot Attention**: Multi-head STEVE-SA with optional RoPE
- **Masking**: Per-slot token masking with configurable ratios
- **Decoders**: MLP, Transformer, or Autoregressive (SPOT-style)

## Project Structure

```
src/
├── slot_attn.py      # Multi-head slot attention (STEVE-SA)
├── slot_mae.py       # Slot-MAE model
├── slot_jepa.py      # Slot-JEPA teacher-student model
├── slot_masks.py     # Per-slot mask generation
├── decoders.py       # Decoder architectures
├── mask_metrics.py   # Evaluation metrics (ARI, IoU, mBO, etc.)
├── data.py           # Dataset loaders
└── utils.py          # Training utilities

train_slot_mae.py     # Slot-MAE training script
train_slot_jepa.py    # Slot-JEPA training script
train_optimized.py    # Shared training utilities
crf.py                # Dense CRF mask refinement
configs/              # YAML training configurations
```

## Usage

### Training Slot-MAE

```bash
python train_slot_mae.py --config configs/dinosaur_coco_slot_mae.yaml
```

### Training Slot-JEPA

```bash
python train_slot_jepa.py --config configs/dinosaur_coco_slot_jepa.yaml
```

### Key Configuration Options

```yaml
slots:
  num_slots: 7          # Number of object slots
  num_iterations: 3     # Slot attention iterations

masking:
  ratio: 0.75           # Fraction of tokens to mask per slot
  min_tokens: 8         # Minimum visible tokens per slot

decoder:
  type: transformer     # Options: mlp, transformer, autoregressive
```

## Datasets

- **COCO**: Place in `data/coco/` with `train2017/` and `val2017/` splits
- **CLEVRTex**: Place in `data/clevrtex/`

## Evaluation Metrics

- **ARI** (Adjusted Rand Index): Clustering quality
- **mIoU**: Mask intersection over union
- **mBO/ABO**: Mean/Average best overlap
- **CorLoc**: Correct localization
- **Boundary IoU**: Mask boundary accuracy

## Dependencies

Core dependencies include PyTorch, einops, wandb, and pydensecrf. External repos (DINOv3, VJEPA2, SPOT) should be cloned into the project root.
