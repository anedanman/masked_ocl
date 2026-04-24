# CRF-Augmented Object-Centric Representation Learning Pipeline

## Overview

This document describes the current training and inference pipeline for object-centric representation learning in this repository after the integration of conditional random fields (CRFs). The system starts from dense DINO patch features, extracts a fixed set of slots with Slot Attention, and trains a slot-conditioned decoder to reconstruct the original DINO feature map. The newly added CRF module refines soft slot assignments using patch-feature similarity in DINO space rather than RGB similarity, and can influence learning in two complementary ways:

1. by directly modifying the slot-attention aggregation weights before the final slot update, and
2. by providing auxiliary supervision targets for both slot assignments and decoder masks.

The implementation is modular and fully configurable through YAML.

## 1. Feature Extraction

Given an input image $x \in \mathbb{R}^{3 \times H \times W}$, a frozen DINO backbone produces a dense patch-feature tensor

$$
F \in \mathbb{R}^{D \times H_f \times W_f},
$$

where $D$ is the feature dimension and $N = H_f W_f$ is the number of patch tokens. We flatten the spatial dimensions to obtain

$$
\mathbf{f}_1, \ldots, \mathbf{f}_N \in \mathbb{R}^D.
$$

These patch features serve two roles:

1. they are the input tokens for Slot Attention, and
2. they define the appearance affinity used by the CRF.

The training target is the original DINO feature map itself, so the decoder learns to reconstruct features rather than pixels.

## 2. Slot Attention Backbone

The slot extractor is a multi-head Slot Attention module with $K$ slots. Let

$$
S^{(t)} = \{\mathbf{s}^{(t)}_1, \ldots, \mathbf{s}^{(t)}_K\}
$$

denote the slots at iteration $t$. At each iteration, the model computes queries from slots and keys/values from DINO features, producing token-to-slot attention logits. After softmax, this yields per-token assignment probabilities

$$
A^{(t)} \in \mathbb{R}^{N \times K},
$$

where each row sums to one across slots.

As in standard Slot Attention, the assignments are then renormalized over tokens for each slot, producing weighted averages of token values that update the slots through a GRU and an MLP.

The current code supports Gaussian, feature-conditioned Gaussian, and k-means-based slot initialization, but the CRF mechanism is independent of the initialization scheme.

## 3. Token-Level CRF in DINO Feature Space

### 3.1 Motivation

The raw slot-attention assignments are locally predicted from slot-feature similarity only. To encourage sharper and more spatially coherent object partitions, we refine these assignments with a dense CRF defined over patch tokens.

Unlike classical image CRFs that use RGB bilateral similarity, this CRF uses DINO patch-feature similarity. This is important because the representation already encodes semantic and part-level structure beyond raw color.

### 3.2 Unary Term

Let $A \in \mathbb{R}^{N \times K}$ be the soft slot assignment distribution produced by Slot Attention for one image. The CRF unary energy is

$$
U_{ik} = -\log A_{ik},
$$

optionally scaled by a unary temperature.

Thus, the CRF starts from Slot Attention’s own beliefs and refines them rather than replacing them with an unrelated segmentation model.

### 3.3 Pairwise Terms

The CRF uses Potts-style smoothing with two possible pairwise kernels.

#### Spatial kernel

A purely spatial kernel encourages nearby tokens to take similar labels:

$$
K^{\text{spatial}}_{ij}
= \exp\left(
- \frac{\lVert \mathbf{p}_i - \mathbf{p}_j \rVert^2}{2\sigma_{\text{sp}}^2}
\right),
$$

where $\mathbf{p}_i$ and $\mathbf{p}_j$ are 2D token coordinates.

#### Appearance kernel in DINO space

The appearance kernel uses either cosine-derived distance or Euclidean distance between DINO patch features:

$$
K^{\text{app}}_{ij}
= \exp\left(
- \frac{d(\mathbf{f}_i,\mathbf{f}_j)}{2\sigma_{\text{app}}^2}
\right)
\exp\left(
- \frac{\lVert \mathbf{p}_i - \mathbf{p}_j \rVert^2}{2\sigma_{\text{app,xy}}^2}
\right).
$$

This combines semantic similarity in feature space with a spatial locality prior.

Both kernels are row-normalized. Optionally, the graph can be sparsified by keeping only the top-$k$ neighbors per token.

### 3.4 Label Compatibility

By default, the CRF uses Potts compatibility:

$$
\mu(k,l)=\mathbb{1}[k \ne l],
$$

so neighboring tokens are penalized when they place mass on different slots. This is the classical dense-CRF choice and remains the default through:

```yaml
crf:
  compatibility:
    type: potts
```

The second experiment iteration adds an optional learned slot-conditioned compatibility. In this mode, each current slot vector is projected with a small MLP, normalized, and compared by cosine similarity:

$$
\mathbf{z}_k =
\operatorname{norm}(\operatorname{MLP}(\mathbf{s}_k)),
\qquad
c_{kl} = \mathbf{z}_k^\top \mathbf{z}_l.
$$

The default transform converts similarity into a non-negative disagreement cost:

$$
\mu(k,l)=\frac{1-c_{kl}}{2\tau},
\qquad
\mu(k,k)=0.
$$

Intuitively, two slots whose embeddings look similar are treated as less mutually incompatible, while dissimilar slots receive stronger pairwise repulsion. This is configured entirely inside the CRF block:

```yaml
crf:
  compatibility:
    type: cosine_mlp
    hidden_dim: 512
    projection_dim: 128
    transform: one_minus_cosine
    temperature: 1.0
    detach_slots: false
    symmetrize: true
    diagonal: zero
```

Supported transforms are `one_minus_cosine`, `cosine`, `negative_cosine`, and `softplus_negative_cosine`. The learned compatibility path also logs compatibility summary statistics, including off-diagonal mean and standard deviation, so runs can reveal whether the layer learns a meaningful non-Potts structure.

### 3.5 Mean-Field Refinement

Starting from $Q^{(0)} = A$, the CRF performs iterative mean-field updates:

$$
Q^{(m+1)} = \operatorname{softmax}\left(
-U - \Psi(Q^{(m)})
\right),
$$

where $\Psi(\cdot)$ is the sum of pairwise messages from the enabled kernels under the configured label compatibility. Intuitively, the CRF increases confidence when semantically similar nearby tokens already agree, and suppresses isolated inconsistent assignments.

With learned compatibility, the Potts message

$$
\sum_l \mathbb{1}[k \ne l] m_l
$$

is replaced with

$$
\sum_l \mu(k,l)m_l,
$$

where $\mu$ is computed from the current slots.

The output is a refined assignment distribution

$$
\widetilde{A} \in \mathbb{R}^{N \times K}.
$$

## 4. How the CRF Interacts with Slot Attention

The current implementation exposes three modes for integrating the CRF into Slot Attention.

### 4.1 `disabled`

The CRF is computed only if needed for auxiliary supervision, but it does not alter the forward slot update.

### 4.2 `replace`

The refined assignments $\widetilde{A}$ are converted back into per-head attention weights and used for the final token aggregation that updates the slots. In this mode, the CRF acts as a structured refinement step before the ultimate slot aggregation.

### 4.3 `blend`

The effective assignment is a convex combination of raw and refined attention:

$$
A^{\star} = (1-\alpha) A + \alpha \widetilde{A},
$$

where $\alpha \in [0,1]$ is configurable.

### 4.4 Final-only vs every-iteration CRF

The CRF can be applied only at the last Slot Attention iteration or at every iteration. Final-only refinement is cheaper and leaves the iterative optimization mostly unchanged. Every-iteration refinement is stronger but more expensive and more interventionist.

### 4.5 Straight-through gradient option

When enabled, the model can use CRF-refined forward values while preserving the gradient pathway of the original slot-attention logits through a straight-through estimator. This lets the CRF influence the forward slot update without making optimization depend entirely on a detached external refinement.

The second experiment iteration adds a `ste_grad_scale` knob:

```yaml
crf:
  slot_attention:
    ste_grad: true
    ste_grad_scale: 0.25
```

This keeps the CRF-refined forward value, but scales the raw-attention backward substitute. The motivation is empirical: the first CRF sweep showed that full-strength STE variants performed much worse than ordinary differentiable CRF replacement. One likely cause is a mismatch between the forward fixed point used for slot updates and the raw Slot Attention gradient used for learning.

The same section now also exposes two stop-gradient controls:

```yaml
crf:
  slot_attention:
    detach_refined: false
    detach_refined_except_final: false
```

`detach_refined: true` treats the CRF as a forward-only optimizer: refined assignments are used to aggregate tokens into slots, but gradients do not pass through the CRF refinement itself. `detach_refined_except_final: true` is more selective: when CRF is applied at every Slot Attention iteration, intermediate CRF refinements are detached while the final refinement remains differentiable. This mirrors the spirit of the Slot Attention fixed-point truncation trick and tests whether repeated differentiable CRF updates destabilize learning.

## 5. Slot-Conditioned Feature Decoder

After slot extraction, the decoder reconstructs the DINO feature map conditioned on the slots.

In the main experimental setup based on `configs/ar_coco.yaml`, the decoder is autoregressive. It predicts feature tokens in a chosen order and uses cross-attention from decoder tokens to slots. The decoder also exposes slot-wise mask estimates derived from its cross-attention weights, which are interpreted as decoder-side object assignments.

Thus the pipeline produces two assignment structures:

1. Slot Attention assignments over DINO tokens.
2. Decoder cross-attention masks over the reconstructed token sequence.

These two structures can now both be related to the CRF-refined assignment target.

## 6. Training Objective

### 6.1 Reconstruction loss

The primary objective remains DINO feature reconstruction:

$$
\mathcal{L}_{\text{rec}} =
\frac{1}{|\Omega|}
\sum_{i \in \Omega}
\lVert \hat{\mathbf{f}}_i - \mathbf{f}_i \rVert_2^2,
$$

where $\Omega$ is the set of predicted tokens.

### 6.2 Optional slot-attention guidance loss

If enabled, the raw Slot Attention assignments are matched to the CRF-refined assignments:

$$
\mathcal{L}_{\text{sa-crf}} =
\mathcal{D}(A,\widetilde{A}),
$$

where $\mathcal{D}$ can be KL divergence, soft cross-entropy, BCE, or MSE, depending on configuration.

This encourages the slot extractor itself to internalize the CRF’s structured prior instead of relying on the CRF only at inference time.

The guidance target can now be temperature-softened before matching:

```yaml
crf:
  guidance:
    slot_attention:
      enabled: true
      loss_type: soft_ce
      lambda_end: 0.005
      lambda_warmup_steps: 20000
      lambda_ramp_steps: 80000
      start_step: 20000
      target_temperature: 2.0
      pred_temperature: 1.0
```

This is meant to avoid forcing the slot extractor to chase an overconfident moving pseudo-label too early in training. `start_step` skips the guidance loss entirely before the configured step; the lambda warmup then controls how quickly the active loss reaches its final weight.

### 6.3 Optional decoder guidance loss

The decoder’s cross-attention masks are also compared to the CRF target:

$$
\mathcal{L}_{\text{dec-crf}} =
\mathcal{D}(M_{\text{dec}}, \widetilde{A}_{\text{img}}),
$$

where $M_{\text{dec}}$ denotes decoder masks and $\widetilde{A}_{\text{img}}$ is the CRF-refined assignment reshaped into per-slot spatial masks.

This supervision does not directly bias cross-attention logits in the current implementation. Instead, it trains the decoder to align its own emergent slot-token correspondence with the CRF-refined object partition.

Decoder guidance supports the same target and prediction temperatures as slot-attention guidance. The second sweep tests much weaker decoder guidance because the first sweep showed that direct decoder supervision against hard CRF targets can dominate reconstruction and harm object-centric metrics.

### 6.4 Existing slot/decoder mask matching loss

The earlier mask-matching objective between Slot Attention masks and decoder masks remains available. Therefore the full training loss can be written as

$$
\mathcal{L}
=
\mathcal{L}_{\text{rec}}
 \lambda_{\text{match}} \mathcal{L}_{\text{match}}
 \lambda_{\text{sa}} \mathcal{L}_{\text{sa-crf}}
 \lambda_{\text{dec}} \mathcal{L}_{\text{dec-crf}}.
$$

Each coefficient can be ramped over training.

## 7. Inference-Time Behavior

At inference time, the active behavior depends on configuration:

1. If CRF slot refinement is disabled, inference follows the original Slot Attention and decoder path.
2. If CRF slot refinement is enabled in `replace` or `blend` mode, the refined assignments affect the extracted slots directly.
3. If only CRF guidance losses were enabled during training, inference remains architecturally unchanged, but the learned model may still produce better object-centric masks because CRF supervision shaped the training dynamics.

The current implementation always computes CRF refinement from DINO patch features and slot-assignment probabilities, never from RGB pixels.

## 8. Configurable Components

The CRF integration is controlled by the `crf` section in the YAML config. The main groups are:

- `crf.enabled`: global on/off switch.
- `crf.num_iterations`: mean-field iterations.
- `crf.spatial.*`: spatial kernel strength and bandwidth.
- `crf.appearance.*`: DINO-feature kernel strength, bandwidth, and spatial extent.
- `crf.compatibility.*`: Potts or learned slot-conditioned label compatibility.
- `crf.pairwise_topk`: optional sparse neighborhood size.
- `crf.slot_attention.*`: how CRF refinement modifies slot updates.
- `crf.guidance.slot_attention.*`: auxiliary loss on Slot Attention assignments.
- `crf.guidance.decoder.*`: auxiliary loss on decoder masks.

The base AR experiment is still defined by `configs/ar_coco.yaml`; the CRF sweeps modify only CRF-related settings.

## 9. Experimental Protocol

To support systematic ablations, the repository includes an experiment generator and launcher:

- `scripts/run_crf_experiments.py`

This script:

1. generates CRF-focused variants of the AR COCO config,
2. launches training through `train_mar.py`,
3. collects per-run summaries, and
4. maintains a leaderboard for easy comparison.

The default sweep includes:

- no-CRF baseline,
- direct CRF replacement,
- CRF blending,
- final-only vs every-iteration CRF,
- straight-through variants,
- slot-only guidance,
- decoder-only guidance,
- combined guidance,
- spatial-only and appearance-only ablations,
- kernel sharpness ablations,
- sparse top-$k$ CRF variants,
- longer mean-field schedules.

The repository also includes a second-round generator:

- `scripts/run_crf_experiments_iter2.py`

This script writes configs to `configs/future_runs_iter2` and maintains a separate summary under `runs/slot-ar/_crf_iter2_summary`. It keeps controls from the first sweep and adds experiments motivated by the first leaderboard:

- sparse all-iteration replacement with top-32 and top-64 neighborhoods,
- gentler CRF weights and fewer mean-field iterations,
- lower unary temperature to trust Slot Attention more strongly,
- high-blend all-iteration CRF,
- reduced-scale STE variants,
- stop-gradient and final-gradient-only CRF variants,
- weak, delayed, temperature-softened guidance losses,
- learned slot-conditioned compatibility variants under `crf.compatibility`, including different projection sizes, compatibility temperatures, sparse graphs, and gradient-control settings.

The iter2 leaderboard includes CRF diagnostics such as assignment delta, entropy, confidence, and compatibility statistics. These are intended to make it easier to distinguish "metric improved because the CRF sharpened good slots" from "metric collapsed because the CRF drove assignments to degenerate confident partitions."

## 10. Summary

The CRF-augmented pipeline now implements a structured refinement layer between raw Slot Attention assignments and final slot aggregation. Its key property is that pairwise consistency is computed in DINO feature space, making the refinement semantically informed rather than purely photometric.

The design supports two complementary hypotheses:

1. structured assignment refinement improves slot extraction directly, and
2. CRF-refined assignments provide useful self-supervision targets for both the slot extractor and the slot-conditioned decoder.

As implemented, the method remains modular, inexpensive to ablate, and directly compatible with the existing object-centric reconstruction framework.
