import torch

from src.models.ar import SlotARDecoder
from src.models.slot_attn import MultiHeadSTEVESA
from src.models.token_crf import TokenFeatureCRF
from src.training.losses import compute_distribution_matching_loss


def test_token_feature_crf_preserves_distribution_shape_and_normalization() -> None:
    crf = TokenFeatureCRF(
        num_iterations=3,
        spatial_weight=2.0,
        spatial_sigma=1.0,
        appearance_weight=4.0,
        appearance_sigma=0.5,
        appearance_spatial_sigma=2.0,
    )
    features = torch.randn(2, 8, 4, 4)
    probs = torch.softmax(torch.randn(2, 16, 3), dim=-1)
    context = crf.build_context(features)
    refined = crf.refine(probs, context)

    assert refined.refined_probs.shape == probs.shape
    assert torch.allclose(
        refined.refined_probs.sum(dim=-1),
        torch.ones_like(refined.refined_probs[..., 0]),
        atol=1e-5,
        rtol=1e-5,
    )
    assert "delta_l1" in refined.stats


def test_token_feature_crf_supports_learned_slot_compatibility() -> None:
    crf = TokenFeatureCRF(
        num_iterations=2,
        spatial_weight=1.0,
        spatial_sigma=1.0,
        appearance_weight=1.0,
        appearance_sigma=0.5,
        appearance_spatial_sigma=2.0,
        slot_size=8,
        compatibility_type="cosine_mlp",
        compatibility_projection_dim=4,
    )
    features = torch.randn(2, 8, 4, 4)
    slots = torch.randn(2, 3, 8)
    probs = torch.softmax(torch.randn(2, 16, 3), dim=-1)
    context = crf.build_context(features)
    refined = crf.refine(probs, context, slot_embeddings=slots)

    assert refined.refined_probs.shape == probs.shape
    assert torch.allclose(
        refined.refined_probs.sum(dim=-1),
        torch.ones_like(refined.refined_probs[..., 0]),
        atol=1e-5,
        rtol=1e-5,
    )
    assert "compatibility_offdiag_mean" in refined.stats


def test_token_feature_crf_supports_transformer_product_compatibility() -> None:
    crf = TokenFeatureCRF(
        num_iterations=2,
        spatial_weight=1.0,
        spatial_sigma=1.0,
        appearance_weight=1.0,
        appearance_sigma=0.5,
        appearance_spatial_sigma=2.0,
        slot_size=16,
        compatibility_type="transformer_product",
        compatibility_hidden_dim=16,
        compatibility_projection_dim=8,
        compatibility_num_heads=4,
        compatibility_transform="softplus_product",
    )
    assert crf.compatibility_output_norm == "l2"
    features = torch.randn(2, 8, 4, 4)
    slots = torch.randn(2, 3, 16)
    probs = torch.softmax(torch.randn(2, 16, 3), dim=-1)
    context = crf.build_context(features)
    refined = crf.refine(probs, context, slot_embeddings=slots)

    assert refined.refined_probs.shape == probs.shape
    assert torch.allclose(
        refined.refined_probs.sum(dim=-1),
        torch.ones_like(refined.refined_probs[..., 0]),
        atol=1e-5,
        rtol=1e-5,
    )
    assert "compatibility_offdiag_mean" in refined.stats


def test_slot_attention_returns_crf_info_when_enabled() -> None:
    slot_attn = MultiHeadSTEVESA(
        num_iterations=3,
        num_slots=4,
        num_heads=1,
        input_size=8,
        out_size=8,
        slot_size=8,
        mlp_hidden_size=16,
        token_crf_cfg={
            "enabled": True,
            "num_iterations": 2,
            "spatial": {"enabled": True, "weight": 1.5, "sigma": 1.0},
            "appearance": {
                "enabled": True,
                "weight": 3.0,
                "sigma": 0.4,
                "spatial_sigma": 2.0,
                "similarity": "cosine",
                "normalize_features": True,
            },
            "slot_attention": {
                "mode": "replace",
                "apply_every_iteration": False,
                "ste_grad": True,
                "return_refined_attn": True,
            },
        },
    )

    feats = torch.randn(2, 8, 4, 4)
    slots, attn_vis, init_loss, info = slot_attn(feats, return_info=True)

    assert slots.shape == (2, 4, 8)
    assert attn_vis.shape == (2, 1, 16, 4)
    assert init_loss is None
    assert info["refined_assignments"] is not None
    assert info["raw_assignments"] is not None
    assert torch.allclose(
        attn_vis.sum(dim=(1, 3)),
        torch.ones_like(attn_vis[:, 0, :, 0]),
        atol=1e-5,
        rtol=1e-5,
    )


def test_slot_ar_decoder_can_apply_cross_attention_crf() -> None:
    decoder = SlotARDecoder(
        slot_size=16,
        feat_dim=8,
        model_dim=24,
        decoder_depth=2,
        num_heads=4,
        register_slots=2,
        token_crf_cfg={
            "enabled": True,
            "num_iterations": 2,
            "spatial": {"enabled": True, "weight": 1.0, "sigma": 1.0},
            "appearance": {
                "enabled": True,
                "weight": 1.0,
                "sigma": 0.5,
                "spatial_sigma": 2.0,
            },
            "slot_attention": {"mode": "disabled"},
            "cross_attention": {"mode": "replace", "apply_all_layers": True},
        },
    )
    feats = torch.randn(2, 8, 4, 4)
    slots = torch.randn(2, 3, 16)
    attn_vis = torch.softmax(torch.randn(2, 1, 16, 3), dim=-1)
    order = torch.arange(16).unsqueeze(0).expand(2, -1)

    output = decoder(feats, slots, attn_vis, order=order)

    assert output.predictions.shape == (2, 16, 8)
    assert output.decoder_masks is not None
    assert output.decoder_masks.shape == (2, 3, 1, 4, 4)
    cross_info = output.info["cross_attention"]
    assert cross_info["raw_assignments"].shape == (2, 16, 3)
    assert cross_info["refined_assignments"].shape == (2, 16, 3)
    assert "delta_l1" in cross_info["crf_stats"]


def test_distribution_matching_supports_mse_alias() -> None:
    pred = torch.tensor([[[0.7, 0.3], [0.1, 0.9]]], dtype=torch.float32)
    target = torch.tensor([[[0.8, 0.2], [0.2, 0.8]]], dtype=torch.float32)
    loss = compute_distribution_matching_loss(pred, target, loss_type="mse", normalize_dim=-1)
    assert loss.item() > 0.0
