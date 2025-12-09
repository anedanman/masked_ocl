"""Metrics related to the evaluation of masks. Taken from https://github.com/amazon-science/object-centric-learning-framework/blob/main/ocl/metrics/masks.py"""
"""Metrics related to the evaluation of masks."""
from typing import Dict, Optional, Sequence, Tuple
import math

import scipy.optimize
import torch
import torch.nn.functional as F
import torchmetrics

from src.utils import adjusted_rand_index, fg_adjusted_rand_index, tensor_to_one_hot


def _convert_predictions_to_masks(prediction: torch.Tensor) -> torch.Tensor:
    if prediction.ndim != 4:
        raise ValueError(f"Expected prediction with shape (B, C, H, W). Got {prediction.shape}.")
    indices = torch.argmax(prediction, dim=1)
    one_hot = torch.nn.functional.one_hot(indices, num_classes=prediction.shape[1])
    return one_hot.permute(0, 3, 1, 2).to(prediction.dtype)


def _binarize_target_masks(target: torch.Tensor) -> torch.Tensor:
    if target.ndim != 4:
        raise ValueError(f"Expected target with shape (B, K, H, W). Got {target.shape}.")
    return (target > 0).to(target.dtype)


def _pairwise_iou_matrix(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    iou_empty: float = 0.0,
) -> torch.Tensor:
    if prediction.numel() == 0 or target.numel() == 0:
        return prediction.new_zeros(
            (prediction.shape[0], target.shape[0]), dtype=torch.float64
        )

    pred = prediction.to(torch.float64)
    tgt = target.to(torch.float64)
    intersection = torch.matmul(pred, tgt.transpose(0, 1))
    pred_area = pred.sum(dim=1, keepdim=True)
    tgt_area = tgt.sum(dim=1, keepdim=True).transpose(0, 1)
    union = pred_area + tgt_area - intersection
    pairwise_iou = intersection / union
    pairwise_iou[union == 0] = iou_empty
    return pairwise_iou


def _match_pairwise_iou(
    pairwise_iou: torch.Tensor,
    matching: str,
    valid_gt: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = pairwise_iou.device
    empty_val = torch.empty(0, dtype=torch.float64, device=device)
    empty_idx = torch.empty(0, dtype=torch.int64, device=device)

    if pairwise_iou.numel() == 0 or pairwise_iou.shape[1] == 0:
        return empty_val, empty_idx, empty_idx

    if matching == "hungarian":
        pred_idxs, true_idxs = scipy.optimize.linear_sum_assignment(
            pairwise_iou.cpu(), maximize=True
        )
        pred_idxs = torch.as_tensor(pred_idxs, dtype=torch.int64, device=device)
        true_idxs = torch.as_tensor(true_idxs, dtype=torch.int64, device=device)
        matched = pairwise_iou[pred_idxs, true_idxs]
        return matched, pred_idxs, true_idxs

    if matching == "best_overlap":
        if valid_gt is None:
            valid_gt = torch.ones(pairwise_iou.shape[1], dtype=torch.bool, device=device)
        else:
            valid_gt = valid_gt.to(dtype=torch.bool, device=device)
        gt_indices = torch.arange(pairwise_iou.shape[1], device=device)[valid_gt]
        if gt_indices.numel() == 0:
            return empty_val, empty_idx, empty_idx
        reduced = pairwise_iou[:, gt_indices]
        best_pred = torch.argmax(reduced, dim=0)
        matched = reduced[best_pred, torch.arange(reduced.shape[1], device=device)]
        return matched, best_pred, gt_indices

    raise ValueError(f"Unknown matching '{matching}'.")


class ARIMetric(torchmetrics.Metric):
    """Computes ARI metric."""

    def __init__(
        self,
        foreground: bool = True,
        convert_target_one_hot: bool = False,
        ignore_overlaps: bool = False,
    ):
        super().__init__()
        self.foreground = foreground
        self.convert_target_one_hot = convert_target_one_hot
        self.ignore_overlaps = ignore_overlaps
        self.add_state(
            "values", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self, prediction: torch.Tensor, target: torch.Tensor, ignore: Optional[torch.Tensor] = None
    ):
        """Update this metric.

        Args:
            prediction: Predicted mask of shape (B, C, H, W) or (B, F, C, H, W), where C is the
                number of classes.
            target: Ground truth mask of shape (B, K, H, W) or (B, F, K, H, W), where K is the
                number of classes.
            ignore: Ignore mask of shape (B, 1, H, W) or (B, 1, K, H, W)
        """
        if prediction.ndim == 5:
            # Merge frames, height and width to single dimension.
            prediction = prediction.transpose(1, 2).flatten(-3, -1)
            target = target.transpose(1, 2).flatten(-3, -1)
            if ignore is not None:
                ignore = ignore.to(torch.bool).transpose(1, 2).flatten(-3, -1)
        elif prediction.ndim == 4:
            # Merge height and width to single dimension.
            prediction = prediction.flatten(-2, -1)
            target = target.flatten(-2, -1)
            if ignore is not None:
                ignore = ignore.to(torch.bool).flatten(-2, -1)
        else:
            raise ValueError(f"Incorrect input shape: f{prediction.shape}")

        if self.ignore_overlaps:
            overlaps = (target > 0).sum(1, keepdim=True) > 1
            if ignore is None:
                ignore = overlaps
            else:
                ignore = ignore | overlaps

        if ignore is not None:
            assert ignore.ndim == 3 and ignore.shape[1] == 1
            prediction = prediction.clone()
            prediction[ignore.expand_as(prediction)] = 0
            target = target.clone()
            target[ignore.expand_as(target)] = 0

        # Make channels / gt labels the last dimension.
        prediction = prediction.transpose(-2, -1)
        target = target.transpose(-2, -1)

        if self.convert_target_one_hot:
            target_oh = tensor_to_one_hot(target, dim=2)
            # For empty pixels (all values zero), one-hot assigns 1 to the first class, correct for
            # this (then it is technically not one-hot anymore).
            target_oh[:, :, 0][target.sum(dim=2) == 0] = 0
            target = target_oh

        # Should be either 0 (empty, padding) or 1 (single object).
        assert torch.all(target.sum(dim=-1) < 2), "Issues with target format, mask non-exclusive"

        if self.foreground:
            ari = fg_adjusted_rand_index(prediction, target)
        else:
            ari = adjusted_rand_index(prediction, target)
        ari = ari.to(self.values.device)

        self.values += ari.sum()
        self.total += len(ari)

    def compute(self) -> torch.Tensor:
        return self.values / self.total


class PatchARIMetric(ARIMetric):
    """Computes ARI metric assuming patch masks as input."""

    def __init__(
        self,
        foreground=True,
        resize_masks_mode: str = "bilinear",
        **kwargs,
    ):
        super().__init__(foreground=foreground, **kwargs)
        self.resize_masks_mode = resize_masks_mode

    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        """Update this metric.

        Args:
            prediction: Predicted mask of shape (B, C, P) or (B, F, C, P), where C is the
                number of classes and P the number of patches.
            target: Ground truth mask of shape (B, K, H, W) or (B, F, K, H, W), where K is the
                number of classes.
        """
        h, w = target.shape[-2:]
        assert h == w

        prediction_resized = resize_patches_to_image(
            prediction, size=h, resize_mode=self.resize_masks_mode
        )

        return super().update(prediction=prediction_resized, target=target)


class UnsupervisedMaskIoUMetric(torchmetrics.Metric):
    """Computes IoU metric for segmentation masks when correspondences to ground truth are not known.

    Uses Hungarian matching to compute the assignment between predicted classes and ground truth
    classes.

    Args:
        use_threshold: If `True`, convert predicted class probabilities to mask using a threshold.
            If `False`, class probabilities are turned into mask using a softmax instead.
        threshold: Value to use for thresholding masks.
        matching: Approach to match predicted to ground truth classes. For "hungarian", computes
            assignment that maximizes total IoU between all classes. For "best_overlap", uses the
            predicted class with maximum overlap for each ground truth class. Using "best_overlap"
            leads to the "average best overlap" metric.
        compute_discovery_fraction: Instead of the IoU, compute the fraction of ground truth classes
            that were "discovered", meaning that they have an IoU greater than some threshold.
        correct_localization: Instead of the IoU, compute the fraction of images on which at least
            one ground truth class was correctly localised, meaning that they have an IoU
            greater than some threshold.
        discovery_threshold: Minimum IoU to count a class as discovered/correctly localized.
        ignore_background: If true, assume class at index 0 of ground truth masks is background class
            that is removed before computing IoU.
        ignore_overlaps: If true, remove points where ground truth masks has overlappign classes from
            predictions and ground truth masks.
    """

    def __init__(
        self,
        use_threshold: bool = False,
        threshold: float = 0.5,
        matching: str = "hungarian",
        compute_discovery_fraction: bool = False,
        correct_localization: bool = False,
        discovery_threshold: float = 0.5,
        ignore_background: bool = False,
        ignore_overlaps: bool = False,
    ):
        super().__init__()
        self.use_threshold = use_threshold
        self.threshold = threshold
        self.discovery_threshold = discovery_threshold
        self.compute_discovery_fraction = compute_discovery_fraction
        self.correct_localization = correct_localization
        if compute_discovery_fraction and correct_localization:
            raise ValueError(
                "Only one of `compute_discovery_fraction` and `correct_localization` can be enabled."
            )

        matchings = ("hungarian", "best_overlap")
        if matching not in matchings:
            raise ValueError(f"Unknown matching type {matching}. Valid values are {matchings}.")
        self.matching = matching
        self.ignore_background = ignore_background
        self.ignore_overlaps = ignore_overlaps

        self.add_state(
            "values", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self, prediction: torch.Tensor, target: torch.Tensor, ignore: Optional[torch.Tensor] = None
    ):
        """Update this metric.

        Args:
            prediction: Predicted mask of shape (B, C, H, W) or (B, F, C, H, W), where C is the
                number of classes. Assumes class probabilities as inputs.
            target: Ground truth mask of shape (B, K, H, W) or (B, F, K, H, W), where K is the
                number of classes.
            ignore: Ignore mask of shape (B, 1, H, W) or (B, 1, K, H, W)
        """
        if prediction.ndim == 5:
            # Merge frames, height and width to single dimension.
            predictions = prediction.transpose(1, 2).flatten(-3, -1)
            targets = target.transpose(1, 2).flatten(-3, -1)
            if ignore is not None:
                ignore = ignore.to(torch.bool).transpose(1, 2).flatten(-3, -1)
        elif prediction.ndim == 4:
            # Merge height and width to single dimension.
            predictions = prediction.flatten(-2, -1)
            targets = target.flatten(-2, -1)
            if ignore is not None:
                ignore = ignore.to(torch.bool).flatten(-2, -1)
        else:
            raise ValueError(f"Incorrect input shape: f{prediction.shape}")

        if self.use_threshold:
            predictions = predictions > self.threshold
        else:
            indices = torch.argmax(predictions, dim=1)
            predictions = torch.nn.functional.one_hot(indices, num_classes=predictions.shape[1])
            predictions = predictions.transpose(1, 2)

        if self.ignore_background:
            targets = targets[:, 1:]

        targets = targets > 0  # Ensure masks are binary

        if self.ignore_overlaps:
            overlaps = targets.sum(1, keepdim=True) > 1
            if ignore is None:
                ignore = overlaps
            else:
                ignore = ignore | overlaps

        if ignore is not None:
            assert ignore.ndim == 3 and ignore.shape[1] == 1
            predictions[ignore.expand_as(predictions)] = 0
            targets[ignore.expand_as(targets)] = 0

        # Should be either 0 (empty, padding) or 1 (single object).
        assert torch.all(targets.sum(dim=1) < 2), "Issues with target format, mask non-exclusive"

        for pred, target in zip(predictions, targets):
            nonzero_classes = torch.sum(target, dim=-1) > 0
            target = target[nonzero_classes]  # Remove empty (e.g. padded) classes
            if len(target) == 0:
                continue  # Skip elements without any target mask

            iou_per_class = unsupervised_mask_iou(
                pred, target, matching=self.matching, reduction="none"
            )

            if self.compute_discovery_fraction:
                discovered = iou_per_class > self.discovery_threshold
                self.values += discovered.sum() / len(discovered)
            elif self.correct_localization:
                correctly_localized = torch.any(iou_per_class > self.discovery_threshold)
                self.values += correctly_localized.sum()
            else:
                self.values += iou_per_class.mean()
            self.total += 1

    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.zeros_like(self.values)
        else:
            return self.values / self.total


class MaskCorLocMetric(UnsupervisedMaskIoUMetric):
    def __init__(self, **kwargs):
        super().__init__(matching="best_overlap", correct_localization=True, **kwargs)


class AverageBestOverlapMetric(UnsupervisedMaskIoUMetric):
    def __init__(self, **kwargs):
        super().__init__(matching="best_overlap", **kwargs)


class BestOverlapObjectRecoveryMetric(UnsupervisedMaskIoUMetric):
    def __init__(self, **kwargs):
        super().__init__(matching="best_overlap", compute_discovery_fraction=True, **kwargs)


class ForegroundPixelAccuracyMetric(torchmetrics.Metric):
    """Computes foreground pixel accuracy using Hungarian/best-overlap matching."""

    def __init__(
        self,
        reduction: str = "micro",
        matching: str = "hungarian",
        ignore_overlaps: bool = False,
    ):
        super().__init__()
        reduction = reduction.lower()
        if reduction not in {"micro", "macro"}:
            raise ValueError("reduction must be 'micro' or 'macro'.")
        if matching not in {"hungarian", "best_overlap"}:
            raise ValueError("matching must be 'hungarian' or 'best_overlap'.")
        self.reduction = reduction
        self.matching = matching
        self.ignore_overlaps = ignore_overlaps
        if reduction == "micro":
            self.add_state(
                "correct", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
            )
            self.add_state(
                "total", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
            )
        else:
            self.add_state(
                "values", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
            )
            self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        ignore: Optional[torch.Tensor] = None,
    ):
        if prediction.ndim == 5:
            predictions = prediction.transpose(1, 2).flatten(-3, -1)
            targets = target.transpose(1, 2).flatten(-3, -1)
            if ignore is not None:
                ignore = ignore.to(torch.bool).transpose(1, 2).flatten(-3, -1)
        elif prediction.ndim == 4:
            predictions = prediction.flatten(-2, -1)
            targets = target.flatten(-2, -1)
            if ignore is not None:
                ignore = ignore.to(torch.bool).flatten(-2, -1)
        else:
            raise ValueError(f"Incorrect input shape: {prediction.shape}")

        indices = torch.argmax(predictions, dim=1)
        predictions = torch.nn.functional.one_hot(indices, num_classes=predictions.shape[1])
        predictions = predictions.transpose(1, 2)

        targets = targets > 0

        if self.ignore_overlaps:
            overlaps = targets.sum(1, keepdim=True) > 1
            if ignore is None:
                ignore = overlaps
            else:
                ignore = ignore | overlaps

        if ignore is not None:
            assert ignore.ndim == 3 and ignore.shape[1] == 1
            predictions = predictions.clone()
            targets = targets.clone()
            predictions[ignore.expand_as(predictions)] = 0
            targets[ignore.expand_as(targets)] = 0

        for pred, tgt in zip(predictions, targets):
            nonzero = torch.sum(tgt, dim=-1) > 0
            tgt = tgt[nonzero]
            if len(tgt) == 0:
                continue

            pairwise_correct = torch.matmul(
                pred.to(torch.float64), tgt.to(torch.float64).transpose(0, 1)
            )

            tgt_pixels = torch.sum(tgt, dim=1).to(torch.float64)

            if self.matching == "hungarian":
                pred_idx, tgt_idx = scipy.optimize.linear_sum_assignment(
                    pairwise_correct.cpu(), maximize=True
                )
                pred_idx = torch.as_tensor(pred_idx, dtype=torch.int64, device=tgt_pixels.device)
                tgt_idx = torch.as_tensor(tgt_idx, dtype=torch.int64, device=tgt_pixels.device)
            else:
                non_empty = tgt_pixels > 0
                best_pred = torch.argmax(pairwise_correct, dim=0)
                pred_idx = best_pred[non_empty]
                tgt_idx = torch.arange(pairwise_correct.shape[1], device=tgt_pixels.device)[non_empty]

            matched_correct = pairwise_correct[pred_idx, tgt_idx]

            if self.reduction == "micro":
                self.correct += matched_correct.sum()
                self.total += tgt_pixels.sum().clamp_min(1.0)
            else:
                per_class_acc = torch.zeros_like(tgt_pixels, dtype=torch.float64)
                denom = tgt_pixels.clamp_min(1.0)
                per_class_acc[tgt_idx] = matched_correct / denom[tgt_idx]
                self.values += per_class_acc.mean()
                self.count += 1

    def compute(self) -> torch.Tensor:
        if self.reduction == "micro":
            return self.correct / self.total.clamp_min(1.0)
        if self.count == 0:
            return torch.zeros_like(self.values)
        return self.values / self.count


class BoundaryIoUMetric(torchmetrics.Metric):
    """Computes boundary IoU between predicted and ground-truth masks."""

    def __init__(
        self,
        matching: str = "hungarian",
        dilation_ratio: float = 0.02,
        min_dilation: int = 1,
        max_dilation: Optional[int] = None,
        ignore_overlaps: bool = False,
    ):
        super().__init__()
        if matching not in {"hungarian", "best_overlap"}:
            raise ValueError("matching must be 'hungarian' or 'best_overlap'.")
        self.matching = matching
        self.dilation_ratio = float(dilation_ratio)
        self.min_dilation = max(1, int(min_dilation))
        self.max_dilation = max_dilation
        self.ignore_overlaps = ignore_overlaps
        self.add_state(
            "values", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def _effective_dilation(self, height: int, width: int) -> int:
        base = int(round(self.dilation_ratio * max(height, width)))
        value = max(self.min_dilation, base)
        if self.max_dilation is not None:
            value = min(value, int(self.max_dilation))
        return max(1, value)

    def _boundary(self, masks: torch.Tensor) -> torch.Tensor:
        _, _, H, W = masks.shape
        dilation = self._effective_dilation(H, W)
        kernel_size = 2 * dilation + 1
        dilated = F.max_pool2d(masks, kernel_size=kernel_size, stride=1, padding=dilation)
        eroded = 1 - F.max_pool2d(1 - masks, kernel_size=kernel_size, stride=1, padding=dilation)
        return (dilated - eroded).clamp_min(0.0).gt(0).to(masks.dtype)

    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        if prediction.ndim != 4 or target.ndim != 4:
            raise ValueError("BoundaryIoUMetric expects tensors of shape (B, C, H, W).")

        pred_masks = _convert_predictions_to_masks(prediction)
        target_masks = _binarize_target_masks(target)

        if self.ignore_overlaps:
            overlaps = target_masks.sum(dim=1, keepdim=True) > 1
            pred_masks = pred_masks.clone()
            target_masks = target_masks.clone()
            pred_masks[overlaps.expand_as(pred_masks)] = 0
            target_masks[overlaps.expand_as(target_masks)] = 0

        pred_boundaries = self._boundary(pred_masks).flatten(-2)
        target_boundaries = self._boundary(target_masks).flatten(-2)

        batch_size = pred_boundaries.shape[0]
        for b in range(batch_size):
            pred_flat = pred_boundaries[b]
            tgt_flat = target_boundaries[b]

            pred_valid = pred_flat.sum(dim=-1) > 0
            tgt_valid = tgt_flat.sum(dim=-1) > 0
            pred_sel = pred_flat[pred_valid]
            tgt_sel = tgt_flat[tgt_valid]

            if tgt_sel.numel() == 0:
                continue
            if pred_sel.numel() == 0:
                self.total += 1
                continue

            iou_value = unsupervised_mask_iou(
                pred_sel, tgt_sel, matching=self.matching, reduction="mean"
            )
            if isinstance(iou_value, torch.Tensor):
                iou_value = float(iou_value.item())
            self.values += iou_value
            self.total += 1

    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.zeros_like(self.values)
        return self.values / self.total


class SegmentationAPARMetric(torchmetrics.Metric):
    """Computes AP/AR across IoU thresholds for segmentation masks."""

    def __init__(
        self,
        thresholds: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9),
        matching: str = "hungarian",
    ):
        super().__init__()
        if matching not in {"hungarian", "best_overlap"}:
            raise ValueError("matching must be 'hungarian' or 'best_overlap'.")
        if not thresholds:
            raise ValueError("At least one IoU threshold is required.")
        self.thresholds = tuple(sorted(float(t) for t in thresholds))
        self.matching = matching
        size = len(self.thresholds)
        self.add_state(
            "tp", default=torch.zeros(size, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state(
            "fp", default=torch.zeros(size, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state(
            "fn", default=torch.zeros(size, dtype=torch.float64), dist_reduce_fx="sum"
        )

    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        if prediction.ndim != 4 or target.ndim != 4:
            raise ValueError("SegmentationAPARMetric expects tensors of shape (B, C, H, W).")

        pred_masks = _convert_predictions_to_masks(prediction).flatten(-2)
        target_masks = _binarize_target_masks(target).flatten(-2)

        batch_size = pred_masks.shape[0]
        for b in range(batch_size):
            pred_flat = pred_masks[b]
            tgt_flat = target_masks[b]

            pred_valid = pred_flat.sum(dim=-1) > 0
            tgt_valid = tgt_flat.sum(dim=-1) > 0
            pred_sel = pred_flat[pred_valid]
            tgt_sel = tgt_flat[tgt_valid]

            num_pred = pred_sel.shape[0]
            num_gt = tgt_sel.shape[0]

            if num_pred == 0 and num_gt == 0:
                continue
            if num_gt == 0:
                self.fp += num_pred
                continue
            if num_pred == 0:
                self.fn += num_gt
                continue

            pairwise_iou = _pairwise_iou_matrix(pred_sel, tgt_sel)
            matched_iou, matched_pred_idx, _ = _match_pairwise_iou(
                pairwise_iou, matching=self.matching
            )
            unique_matched_pred = (
                matched_pred_idx.unique().numel() if matched_pred_idx.numel() > 0 else 0
            )
            unmatched_preds = num_pred - unique_matched_pred

            for idx, thr in enumerate(self.thresholds):
                tp = (matched_iou >= thr).sum()
                fp = unmatched_preds + (matched_iou < thr).sum()
                fn = num_gt - tp
                self.tp[idx] += tp
                self.fp[idx] += fp
                self.fn[idx] += fn

    def compute(self) -> Dict[str, float]:
        precision = self.tp / torch.clamp(self.tp + self.fp, min=1.0)
        recall = self.tp / torch.clamp(self.tp + self.fn, min=1.0)
        result: Dict[str, float] = {}
        for thr, prec, rec in zip(self.thresholds, precision.tolist(), recall.tolist()):
            tag = f"{int(round(thr * 100)):02d}"
            result[f"ap@{tag}"] = float(prec)
            result[f"ar@{tag}"] = float(rec)
        result["ap"] = float(precision.mean().item())
        result["ar"] = float(recall.mean().item())
        return result


def unsupervised_mask_iou(
    pred_mask: torch.Tensor,
    true_mask: torch.Tensor,
    matching: str = "hungarian",
    reduction: str = "mean",
    iou_empty: float = 0.0,
) -> torch.Tensor:
    """Compute intersection-over-union (IoU) between masks with unknown class correspondences.

    This metric is also known as Jaccard index. Note that this is a non-batched implementation.

    Args:
        pred_mask: Predicted mask of shape (C, N), where C is the number of predicted classes and
            N is the number of points. Masks are assumed to be binary.
        true_mask: Ground truth mask of shape (K, N), where K is the number of ground truth
            classes and N is the number of points. Masks are assumed to be binary.
        matching: How to match predicted classes to ground truth classes. For "hungarian", computes
            assignment that maximizes total IoU between all classes. For "best_overlap", uses the
            predicted class with maximum overlap for each ground truth class (each predicted class
            can be assigned to multiple ground truth classes). Empty ground truth classes are
            assigned IoU of zero.
        reduction: If "mean", return IoU averaged over classes. If "none", return per-class IoU.
        iou_empty: IoU for the case when a class does not occur, but was also not predicted.

    Returns:
        Mean IoU over classes if reduction is `mean`, tensor of shape (K,) containing per-class IoU
        otherwise.
    """
    assert pred_mask.ndim == 2
    assert true_mask.ndim == 2
    n_gt_classes = len(true_mask)
    pred_mask = pred_mask.unsqueeze(1).to(torch.bool)
    true_mask = true_mask.unsqueeze(0).to(torch.bool)

    intersection = torch.sum(pred_mask & true_mask, dim=-1).to(torch.float64)
    union = torch.sum(pred_mask | true_mask, dim=-1).to(torch.float64)
    pairwise_iou = intersection / union

    # Remove NaN from divide-by-zero: class does not occur, and class was not predicted.
    pairwise_iou[union == 0] = iou_empty

    valid_gt = torch.sum(true_mask.squeeze(0), dim=1) > 0
    matched_iou, pred_idxs, true_idxs = _match_pairwise_iou(
        pairwise_iou, matching=matching, valid_gt=valid_gt
    )

    iou = torch.zeros(n_gt_classes, dtype=torch.float64, device=pairwise_iou.device)
    if matched_iou.numel() > 0:
        iou[true_idxs] = matched_iou

    if reduction == "mean":
        return iou.mean()
    else:
        return iou
    

def resize_patches_to_image(
    patches: torch.Tensor,
    size: Optional[int] = None,
    scale_factor: Optional[float] = None,
    resize_mode: str = "bilinear",
) -> torch.Tensor:
    """Convert and resize a tensor of patches to image shape.

    This method requires that the patches can be converted to a square image.

    Args:
        patches: Patches to be converted of shape (..., C, P), where C is the number of channels and
            P the number of patches.
        size: Image size to resize to.
        scale_factor: Scale factor by which to resize the patches. Can be specified alternatively to
            `size`.
        resize_mode: Method to resize with. Valid options are "nearest", "nearest-exact", "bilinear",
            "bicubic".

    Returns:
        Tensor of shape (..., C, S, S) where S is the image size.
    """
    has_size = size is None
    has_scale = scale_factor is None
    if has_size == has_scale:
        raise ValueError("Exactly one of `size` or `scale_factor` must be specified.")

    n_channels = patches.shape[-2]
    n_patches = patches.shape[-1]
    patch_size_float = math.sqrt(n_patches)
    patch_size = int(math.sqrt(n_patches))
    if patch_size_float != patch_size:
        raise ValueError("The number of patches needs to be a perfect square.")

    image = torch.nn.functional.interpolate(
        patches.view(-1, n_channels, patch_size, patch_size),
        size=size,
        scale_factor=scale_factor,
        mode=resize_mode,
    )

    return image.view(*patches.shape[:-1], image.shape[-2], image.shape[-1])
