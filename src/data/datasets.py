import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Tuple, Dict, List, Optional, Any
import torchvision
from torchvision.transforms import v2
import torchvision.transforms.functional as TF


def _load_coco_annotations(coco_cls, ann_file: str):
    """Load large COCO JSON files with orjson when available.

    The standard-library decoder remains the fallback so existing environments keep
    working, while orjson avoids CPython decoder instability on memory-constrained hosts.
    """
    try:
        import orjson
    except ImportError:
        return coco_cls(ann_file)

    print("loading annotations into memory with orjson...")
    with open(ann_file, "rb") as handle:
        dataset = orjson.loads(handle.read())
    coco = coco_cls()
    coco.dataset = dataset
    coco.createIndex()
    return coco


def make_transform(resize_size: int = 256, crop: bool = False):
    """Create image transform pipeline using torchvision v2.
    
    Args:
        resize_size: Target size for image resizing.
        crop: If True, resize smaller edge to resize_size and center crop.
              If False, resize (squash) to (resize_size, resize_size).
        
    Returns:
        Composed transform pipeline.
    """
    to_tensor = v2.ToImage()
    if crop:
        # Match SPOT behavior: Resize smaller edge -> CenterCrop
        resize = v2.Resize(resize_size, antialias=True)
        crop_op = v2.CenterCrop(resize_size)
        transforms_list = [to_tensor, resize, crop_op]
    else:
        # Original behavior: Squash to square
        resize = v2.Resize((resize_size, resize_size), antialias=True)
        transforms_list = [to_tensor, resize]

    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    transforms_list.extend([to_float, normalize])
    return v2.Compose(transforms_list)


class COCODataset(Dataset):
    """
    COCO dataset wrapper that uses instance segmentation annotations (like COCO2017 from SPOT).

    Uses pycocotools to decode RLE masks from instance annotations, matching the behavior
    of the SPOT evaluation dataset.

    The ``max_objects`` argument filters out images that contain more than the requested number
    of instances / classes but does not truncate the returned masks.
    """

    # Same 81 categories as COCO2017 in spot/datasets.py
    NUM_CLASSES = 81
    CAT_LIST = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
        43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88,
        89, 90
    ]

    def __init__(
        self,
        data_root: str,
        split: str = "train2017",
        *,
        mode: str = "instance",
        max_objects: Optional[int] = 20,
        image_size: int = 256,
        max_samples: Optional[int] = None,
        min_area: float = 0.0,
        return_properties: bool = True,
        return_masks: bool = True,
        properties_from_bboxes: bool = False,
        horizontal_flip_prob: float = 0.0,
        extra_image_dirs: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        if mode not in {"instance", "class"}:
            raise ValueError(f"Unsupported mode '{mode}'. Expected 'instance' or 'class'.")

        # Import pycocotools here to match COCO2017 behavior
        from pycocotools.coco import COCO
        from pycocotools import mask as coco_mask_utils

        self.data_root = data_root
        self.split = split
        self.mode = mode
        self.max_objects = max_objects
        self.image_size = image_size
        self.max_samples = max_samples
        self.min_area = float(min_area)
        self.return_properties = return_properties
        self.return_masks = return_masks
        self.properties_from_bboxes = bool(properties_from_bboxes)
        if self.return_properties and not self.return_masks and not self.properties_from_bboxes:
            raise ValueError("Cannot return properties when masks are disabled (unless properties_from_bboxes=True).")
        self.horizontal_flip_prob = float(horizontal_flip_prob)
        if not 0.0 <= self.horizontal_flip_prob <= 1.0:
            raise ValueError(
                f"horizontal_flip_prob must be between 0 and 1 (got {horizontal_flip_prob})."
            )

        # Use crop=True to match SPOT behavior (Resize smaller edge + CenterCrop)
        self.transform = make_transform(resize_size=image_size, crop=True)

        # Use instance annotations (like COCO2017)
        ann_file = os.path.join(data_root, "annotations", f"instances_{split}.json")
        if not os.path.exists(ann_file):
            raise FileNotFoundError(f"Instance annotations not found at {ann_file}")

        # Find image directory
        self.image_dir = os.path.join(data_root, split)
        if not os.path.isdir(self.image_dir):
            self.image_dir = os.path.join(data_root, "images", split)
            if not os.path.isdir(self.image_dir):
                raise FileNotFoundError(f"Image directory not found at {self.image_dir}")

        # Load COCO annotations
        self.coco = _load_coco_annotations(COCO, ann_file)
        self.coco_mask_utils = coco_mask_utils

        # Use the same category mapping as COCO2017
        self.num_categories = self.NUM_CLASSES
        self.category_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(self.CAT_LIST)}
        self.idx_to_category_id = {idx: cat_id for cat_id, idx in self.category_id_to_idx.items()}
        self.property_dim = self.num_categories + 3  # category one-hot + center_x + center_y + presence

        # Get all image IDs
        self.image_ids = list(self.coco.imgs.keys())

        if max_samples is not None:
            self.image_ids = self.image_ids[:max_samples]

        # Build samples list
        self.samples = []
        for img_id in self.image_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            self.samples.append({
                "image_id": img_id,
                "image_file": img_info["file_name"],
                "width": img_info["width"],
                "height": img_info["height"],
            })

        # Optionally extend with unannotated image directories (e.g. COCO
        # unlabeled2017). These samples carry no annotations, so they are only
        # valid for images-only training.
        self.extra_image_dirs = [str(d) for d in (extra_image_dirs or [])]
        self.num_extra_samples = 0
        if self.extra_image_dirs:
            if self.return_masks or self.return_properties:
                raise ValueError(
                    "extra_image_dirs adds unannotated images; it requires "
                    "return_masks=False and return_properties=False."
                )
            extra_id = 10_000_000  # keep synthetic ids clear of real COCO ids
            for image_dir in self.extra_image_dirs:
                image_dir = os.path.abspath(image_dir)
                if not os.path.isdir(image_dir):
                    raise FileNotFoundError(f"Extra image directory not found at {image_dir}")
                files = sorted(
                    f for f in os.listdir(image_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                )
                if not files:
                    raise FileNotFoundError(f"No images found in extra image directory {image_dir}")
                for name in files:
                    self.samples.append({
                        "image_id": extra_id,
                        "image_file": os.path.join(image_dir, name),
                        "width": 0,
                        "height": 0,
                    })
                    extra_id += 1
                self.num_extra_samples += len(files)

        self.filtered_out = 0

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, sample: Dict[str, Any], horizontal_flip: bool = False) -> torch.Tensor:
        img_path = sample["image_file"]
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.image_dir, img_path)
        img = Image.open(img_path).convert("RGB")
        if horizontal_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return self.transform(img)

    def _resize_mask(self, mask: np.ndarray) -> torch.Tensor:
        # Match SPOT behavior: Resize smaller edge -> CenterCrop
        mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
        mask_img = TF.resize(mask_img, self.image_size, interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        mask_img = TF.center_crop(mask_img, self.image_size)
        mask_resized = np.array(mask_img, dtype=np.uint8) > 0
        return torch.from_numpy(mask_resized.astype(np.float32))

    def _gen_seg_n_insta_masks(self, annotations, h, w):
        seg_mask = np.zeros((h, w), dtype=np.uint8)
        insta_mask = np.zeros((h, w), dtype=np.uint8)
        overlap_count = np.zeros((h, w), dtype=np.uint8)
        category_ids = []

        inst_id = 0
        for ann in annotations:
            cat = ann["category_id"]
            if cat not in self.category_id_to_idx:
                continue

            # Robust decoding for polygons + RLE + crowd
            m = self.coco.annToMask(ann).astype(np.uint8)  # (h, w) in {0,1}
            if m.sum() == 0:
                continue

            inst_id += 1
            category_ids.append(cat)

            c = self.category_id_to_idx[cat]  # O(1), don’t use CAT_LIST.index()

            # fill only where empty (first-come-first-served)
            fill = (m == 1) & (seg_mask == 0)
            seg_mask[fill] = c

            fill_i = (m == 1) & (insta_mask == 0)
            insta_mask[fill_i] = inst_id

            overlap_count += m

        ignore_mask = (overlap_count > 1).astype(np.uint8)
        return seg_mask, insta_mask, ignore_mask, category_ids

    def _build_instance_masks(
        self, insta_mask: np.ndarray, seg_mask: np.ndarray, category_ids: List[int]
    ) -> Tuple[List[torch.Tensor], List[int], List[Dict[str, Any]]]:
        """Build per-instance mask tensors from the combined instance mask."""
        masks: List[torch.Tensor] = []
        categories: List[int] = []
        metadata: List[Dict[str, Any]] = []

        for i, cat_id in enumerate(category_ids, 1):
            mask_bool = (insta_mask == i)
            if not mask_bool.any():
                continue
            masks.append(self._resize_mask(mask_bool))
            categories.append(cat_id)
            metadata.append({
                "mask": mask_bool.copy(),
                "category_id": cat_id,
                "bbox": None,
                "area": float(mask_bool.sum()),
            })

        return masks, categories, metadata

    @staticmethod
    def _bbox_from_mask(mask_bool: np.ndarray, orig_w: int, orig_h: int) -> Tuple[float, float, float, float]:
        ys, xs = np.nonzero(mask_bool)
        if len(xs) == 0:
            return 0.0, 0.0, 0.0, 0.0

        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()

        x = xmin / max(orig_w, 1)
        y = ymin / max(orig_h, 1)
        w = (xmax - xmin + 1) / max(orig_w, 1)
        h = (ymax - ymin + 1) / max(orig_h, 1)
        return float(x), float(y), float(w), float(h)

    def _build_properties(
        self,
        categories: List[int],
        masks: List[torch.Tensor],
        size: int,
    ) -> torch.Tensor:
        num_instances = len(categories)
        rows = self.max_objects if self.max_objects is not None else num_instances
        properties = torch.zeros(rows, self.property_dim, dtype=torch.float32)

        for idx, cat_id in enumerate(categories):
            if idx >= rows:
                break
            if cat_id in self.category_id_to_idx:
                cat_idx = self.category_id_to_idx[cat_id]
                properties[idx, cat_idx] = 1.0

            # Use the resized/cropped mask to compute properties relative to the crop
            mask_bool = masks[idx].numpy() > 0.5
            ys, xs = np.nonzero(mask_bool)
            if len(xs) == 0:
                continue

            xmin, xmax = xs.min(), xs.max()
            ymin, ymax = ys.min(), ys.max()

            # Normalize by crop size
            center_x = ((xmin + xmax + 1) / 2.0) / max(size, 1)
            center_y = ((ymin + ymax + 1) / 2.0) / max(size, 1)

            properties[idx, self.num_categories] = float(center_x)
            properties[idx, self.num_categories + 1] = float(center_y)
            properties[idx, self.num_categories + 2] = 1.0

        return properties

    def _bbox_center_in_crop(
        self,
        bbox: List[float],
        orig_w: int,
        orig_h: int,
    ) -> Tuple[float, float]:
        if orig_w <= 0 or orig_h <= 0:
            return 0.0, 0.0
        x, y, bw, bh = bbox
        if orig_w <= orig_h:
            scale = self.image_size / float(orig_w)
            new_w = self.image_size
            new_h = int(round(orig_h * scale))
            crop_x = 0
            crop_y = max((new_h - self.image_size) // 2, 0)
        else:
            scale = self.image_size / float(orig_h)
            new_h = self.image_size
            new_w = int(round(orig_w * scale))
            crop_y = 0
            crop_x = max((new_w - self.image_size) // 2, 0)

        center_x = (x + 0.5 * bw) * scale - crop_x
        center_y = (y + 0.5 * bh) * scale - crop_y

        center_x = center_x / max(self.image_size, 1)
        center_y = center_y / max(self.image_size, 1)
        return float(center_x), float(center_y)

    def _build_properties_from_bboxes(
        self,
        annotations: List[Dict[str, Any]],
        orig_w: int,
        orig_h: int,
        horizontal_flip: bool,
    ) -> Tuple[torch.Tensor, int]:
        valid: List[Tuple[int, float, float]] = []
        for ann in annotations:
            cat_id = ann.get("category_id", None)
            if cat_id not in self.category_id_to_idx:
                continue
            if ann.get("area", 0.0) < self.min_area:
                continue
            bbox = ann.get("bbox", None)
            if bbox is None:
                continue
            cx, cy = self._bbox_center_in_crop(bbox, orig_w, orig_h)
            if horizontal_flip:
                cx = 1.0 - cx
            cx = float(min(max(cx, 0.0), 1.0))
            cy = float(min(max(cy, 0.0), 1.0))
            valid.append((cat_id, cx, cy))

        rows = self.max_objects if self.max_objects is not None else len(valid)
        properties = torch.zeros(rows, self.property_dim, dtype=torch.float32)
        max_fill = min(rows, len(valid))
        for idx in range(max_fill):
            cat_id, cx, cy = valid[idx]
            cat_idx = self.category_id_to_idx[cat_id]
            properties[idx, cat_idx] = 1.0
            properties[idx, self.num_categories] = cx
            properties[idx, self.num_categories + 1] = cy
            properties[idx, self.num_categories + 2] = 1.0
        return properties, len(valid)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        do_hflip = False
        if self.horizontal_flip_prob > 0.0:
            do_hflip = bool(torch.rand(1).item() < self.horizontal_flip_prob)
        image = self._load_image(sample, horizontal_flip=do_hflip)

        output: Dict[str, torch.Tensor] = {
            "image": image,
            "image_id": torch.tensor(sample["image_id"]),
        }

        if not self.return_masks:
            if self.return_properties:
                img_id = sample["image_id"]
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                annotations = self.coco.loadAnns(ann_ids)
                props, num_instances = self._build_properties_from_bboxes(
                    annotations,
                    sample["width"],
                    sample["height"],
                    do_hflip,
                )
                output["properties"] = props
                output["num_instances"] = torch.tensor(num_instances, dtype=torch.long)
            return output

        # Load annotations using pycocotools (like COCO2017)
        img_id = sample["image_id"]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)

        h, w = sample["height"], sample["width"]

        # Generate masks using the same logic as COCO2017
        seg_mask, insta_mask, ignore_mask, raw_category_ids = self._gen_seg_n_insta_masks(
            annotations, h, w
        )

        # Apply horizontal flip if needed
        if do_hflip:
            seg_mask = np.fliplr(seg_mask).copy()
            insta_mask = np.fliplr(insta_mask).copy()
            ignore_mask = np.fliplr(ignore_mask).copy()

        # Build per-instance masks
        masks_list, category_ids, metadata = self._build_instance_masks(
            insta_mask, seg_mask, raw_category_ids
        )

        num_instances = len(category_ids)

        if self.max_objects is not None:
            mask_tensor = torch.zeros(
                self.max_objects, self.image_size, self.image_size, dtype=torch.float32
            )
            category_tensor = torch.full((self.max_objects,), -1, dtype=torch.long)
            for i, mask in enumerate(masks_list):
                if i >= self.max_objects:
                    break
                mask_tensor[i] = mask
                category_tensor[i] = category_ids[i]
        else:
            mask_tensor = (
                torch.stack(masks_list)
                if masks_list
                else torch.zeros((0, self.image_size, self.image_size), dtype=torch.float32)
            )
            category_tensor = (
                torch.tensor(category_ids, dtype=torch.long)
                if category_ids
                else torch.empty((0,), dtype=torch.long)
            )

        output.update(
            {
                "masks": mask_tensor,
                "categories": category_tensor,
                "num_instances": torch.tensor(num_instances, dtype=torch.long),
            }
        )
        ignore_mask_tensor = self._resize_mask(ignore_mask > 0)
        output["ignore_mask"] = ignore_mask_tensor.unsqueeze(0)

        if self.return_properties:
            if self.properties_from_bboxes:
                props, _ = self._build_properties_from_bboxes(
                    annotations,
                    sample["width"],
                    sample["height"],
                    do_hflip,
                )
            else:
                props = self._build_properties(
                    category_ids, masks_list, self.image_size
                )
            output["properties"] = props

        # Areas and bounding boxes (normalized to [0, 1])
        if self.max_objects is not None:
            areas_tensor = torch.zeros(self.max_objects, dtype=torch.float32)
            bboxes_tensor = torch.zeros(self.max_objects, 4, dtype=torch.float32)
        else:
            areas_tensor = torch.zeros(num_instances, dtype=torch.float32)
            bboxes_tensor = torch.zeros(num_instances, 4, dtype=torch.float32)

        crop_area = self.image_size * self.image_size
        max_iter = min(num_instances, len(masks_list))
        if self.max_objects is not None:
            max_iter = min(max_iter, self.max_objects)
        for i in range(max_iter):
            # Use resized/cropped mask for area and bbox
            mask_tensor = masks_list[i]
            area_px = mask_tensor.sum().item()
            areas_tensor[i] = area_px / crop_area

            x, y, bw, bh = self._bbox_from_mask(
                mask_tensor.numpy() > 0.5, self.image_size, self.image_size
            )
            bboxes_tensor[i, 0] = x
            bboxes_tensor[i, 1] = y
            bboxes_tensor[i, 2] = bw
            bboxes_tensor[i, 3] = bh

        output["areas"] = areas_tensor
        output["bboxes"] = bboxes_tensor

        return output


def get_coco_dataloaders(
    data_root: str,
    train_batch_size: int = 32,
    val_batch_size: Optional[int] = None,
    train_num_workers: int = 4,
    val_num_workers: Optional[int] = None,
    image_size: int = 256,
    max_objects: Optional[int] = 20,
    max_samples_train: Optional[int] = None,
    max_samples_val: Optional[int] = None,
    min_area: float = 0.0,
    return_properties: bool = True,
    train_split: str = "train2017",
    val_split: str = "val2017",
    mode: str = "instance",
    train_return_masks: bool = True,
    val_return_masks: bool = True,
    properties_from_bboxes: bool = False,
    train_horizontal_flip_prob: float = 0.5,
    val_horizontal_flip_prob: float = 0.0,
    train_pin_memory: bool = True,
    train_persistent_workers: Optional[bool] = None,
    train_prefetch_factor: Optional[int] = 2,
    train_extra_image_dirs: Optional[List[str]] = None,
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create COCO dataloaders for train and validation using panoptic annotations.

    Args:
        data_root: Root of the COCO dataset.
        train_batch_size: Batch size for the training dataloader.
        val_batch_size: Optional batch size for validation. Defaults to ``train_batch_size``.
        train_num_workers: Number of dataloader workers for training.
        val_num_workers: Optional number of workers for validation (defaults to ``train_num_workers``).
        image_size: Output image size (square, resized with torchvision v2 pipeline).
        max_objects: Maximum number of objects/classes allowed per image; images with more
            objects are filtered out. Padding uses this value when provided.
        max_samples_train: Optional limit on the number of training samples.
        max_samples_val: Optional limit on the number of validation samples.
        min_area: Minimum segment area (in pixels) to keep.
        return_properties: Whether to return per-mask property vectors.
        train_split: Training split name (e.g. ``train2017``).
        val_split: Validation split name (e.g. ``val2017``).
        mode: ``"instance"`` for instance masks or ``"class"`` for class-level masks.
        train_horizontal_flip_prob: Probability of applying random horizontal flip to train images/masks.
        val_horizontal_flip_prob: Probability of applying random horizontal flip to validation data.
    """
    if val_batch_size is None:
        val_batch_size = train_batch_size
    if val_num_workers is None:
        val_num_workers = train_num_workers

    train_props = return_properties and (train_return_masks or properties_from_bboxes)
    val_props = return_properties and (val_return_masks or properties_from_bboxes)

    train_dataset = COCODataset(
        data_root=data_root,
        split=train_split,
        mode=mode,
        max_objects=max_objects,
        image_size=image_size,
        max_samples=max_samples_train,
        min_area=min_area,
        return_properties=train_props,
        return_masks=train_return_masks,
        properties_from_bboxes=properties_from_bboxes,
        horizontal_flip_prob=train_horizontal_flip_prob,
        extra_image_dirs=train_extra_image_dirs,
    )

    val_dataset = COCODataset(
        data_root=data_root,
        split=val_split,
        mode=mode,
        max_objects=max_objects,
        image_size=image_size,
        max_samples=max_samples_val,
        min_area=min_area,
        return_properties=val_props,
        return_masks=val_return_masks,
        properties_from_bboxes=properties_from_bboxes,
        horizontal_flip_prob=val_horizontal_flip_prob,
    )

    dataloaders = {
        "train": torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=train_num_workers,
            pin_memory=bool(train_pin_memory),
            persistent_workers=(
                train_persistent_workers if train_persistent_workers is not None else train_num_workers > 0
            ),
            prefetch_factor=(
                train_prefetch_factor if train_num_workers > 0 else None
            ),
        ),
        "val": torch.utils.data.DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=val_num_workers,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=None,
        ),
    }

    return dataloaders


class VOCDataset(Dataset):
    """
    Pascal VOC 2012 segmentation dataset.

    Returns a dictionary compatible with the training pipeline:
        - image: normalized image tensor (C, H, W)
        - masks: instance masks tensor (max_objects, H, W)
        - image_id: integer id tensor for the sample
        - categories: semantic class ids for each instance (max_objects,)
    """

    # VOC has 21 classes (20 object classes + background)
    NUM_CLASSES = 21
    CLASS_NAMES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
        'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
        'train', 'tvmonitor'
    ]

    def __init__(
        self,
        data_root: str,
        split: str = "trainaug",
        *,
        max_objects: Optional[int] = 20,
        image_size: int = 256,
        max_samples: Optional[int] = None,
        return_masks: bool = True,
        return_properties: bool = True,
        horizontal_flip_prob: float = 0.0,
    ) -> None:
        """
        Args:
            data_root: Path to VOCdevkit/VOC2012 directory.
            split: 'trainaug', 'train', or 'val'.
            max_objects: Maximum number of objects to pad to.
            image_size: Target image size (square).
            max_samples: Maximum number of samples to load.
            return_masks: Whether to return masks (set False for training).
            return_properties: Whether to return property vectors.
            horizontal_flip_prob: Probability of horizontal flip augmentation.
        """
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.max_objects = max_objects
        self.image_size = image_size
        self.return_masks = return_masks
        self.return_properties = return_properties
        if self.return_properties and not self.return_masks:
            raise ValueError("Cannot return properties when masks are disabled.")
        self.horizontal_flip_prob = float(horizontal_flip_prob)

        # Read image list
        imglist_fp = os.path.join(data_root, 'ImageSets', 'Segmentation', f'{split}.txt')
        if not os.path.exists(imglist_fp):
            raise FileNotFoundError(f"Image list not found at {imglist_fp}")

        with open(imglist_fp, 'r') as f:
            self.imglist = [line.strip() for line in f if line.strip()]

        if max_samples is not None:
            self.imglist = self.imglist[:max_samples]

        # Property dimension: class one-hot + center_x + center_y + presence
        self.property_dim = self.NUM_CLASSES + 3

        # Use crop=True to match SPOT behavior (Resize smaller edge + CenterCrop)
        self.transform = make_transform(resize_size=image_size, crop=True)

    def __len__(self) -> int:
        return len(self.imglist)

    def _load_image(self, imgname: str, horizontal_flip: bool = False) -> torch.Tensor:
        img_path = os.path.join(self.data_root, 'JPEGImages', f'{imgname}.jpg')
        img = Image.open(img_path).convert('RGB')
        if horizontal_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return self.transform(img)

    def _load_mask(self, imgname: str, mask_type: str, horizontal_flip: bool = False) -> np.ndarray:
        """Load segmentation mask (SegmentationClass or SegmentationObject)."""
        mask_path = os.path.join(self.data_root, mask_type, f'{imgname}.png')
        if not os.path.exists(mask_path):
            return None
        mask = Image.open(mask_path)
        if horizontal_flip:
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # Resize with nearest neighbor to preserve labels
        mask = TF.resize(mask, self.image_size, interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        mask = TF.center_crop(mask, self.image_size)
        return np.array(mask)

    def _build_instance_masks(
        self, instance_mask: np.ndarray, class_mask: np.ndarray
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """Build per-instance binary masks from VOC instance segmentation."""
        masks = []
        categories = []

        # Get unique instance IDs (excluding 0=background and 255=boundary)
        instance_ids = np.unique(instance_mask)
        instance_ids = instance_ids[(instance_ids != 0) & (instance_ids != 255)]

        for inst_id in instance_ids:
            mask_bool = (instance_mask == inst_id)
            if not mask_bool.any():
                continue

            # Get the class for this instance from the class mask
            inst_pixels = class_mask[mask_bool]
            inst_pixels = inst_pixels[(inst_pixels != 0) & (inst_pixels != 255)]
            if len(inst_pixels) == 0:
                continue
            class_id = int(np.bincount(inst_pixels).argmax())

            masks.append(torch.from_numpy(mask_bool.astype(np.float32)))
            categories.append(class_id)

        return masks, categories

    def _build_properties(
        self,
        categories: List[int],
        masks: List[torch.Tensor],
    ) -> torch.Tensor:
        """Build property vectors (class one-hot + normalized center + presence)."""
        num_instances = len(categories)
        rows = self.max_objects if self.max_objects is not None else num_instances
        properties = torch.zeros(rows, self.property_dim, dtype=torch.float32)

        for idx, cat_id in enumerate(categories):
            if idx >= rows:
                break
            # Class one-hot
            if 0 <= cat_id < self.NUM_CLASSES:
                properties[idx, cat_id] = 1.0

            # Compute center from mask
            mask_bool = masks[idx].numpy() > 0.5
            ys, xs = np.nonzero(mask_bool)
            if len(xs) == 0:
                continue

            center_x = ((xs.min() + xs.max() + 1) / 2.0) / max(self.image_size, 1)
            center_y = ((ys.min() + ys.max() + 1) / 2.0) / max(self.image_size, 1)

            properties[idx, self.NUM_CLASSES] = float(center_x)
            properties[idx, self.NUM_CLASSES + 1] = float(center_y)
            properties[idx, self.NUM_CLASSES + 2] = 1.0  # presence

        return properties

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        imgname = self.imglist[idx]

        do_hflip = False
        if self.horizontal_flip_prob > 0.0:
            do_hflip = bool(torch.rand(1).item() < self.horizontal_flip_prob)

        image = self._load_image(imgname, horizontal_flip=do_hflip)

        output: Dict[str, torch.Tensor] = {
            'image': image,
            'image_id': torch.tensor(idx),
        }

        if not self.return_masks:
            return output

        # Load masks
        instance_mask = self._load_mask(imgname, 'SegmentationObject', horizontal_flip=do_hflip)
        class_mask = self._load_mask(imgname, 'SegmentationClass', horizontal_flip=do_hflip)

        if instance_mask is None or class_mask is None:
            # Return empty masks if not available
            if self.max_objects is not None:
                output['masks'] = torch.zeros(self.max_objects, self.image_size, self.image_size)
                output['categories'] = torch.full((self.max_objects,), -1, dtype=torch.long)
            else:
                output['masks'] = torch.zeros(0, self.image_size, self.image_size)
                output['categories'] = torch.empty(0, dtype=torch.long)
            output['num_instances'] = torch.tensor(0, dtype=torch.long)
            output['ignore_mask'] = torch.zeros(1, self.image_size, self.image_size)
            if self.return_properties:
                rows = self.max_objects if self.max_objects is not None else 0
                output['properties'] = torch.zeros(rows, self.property_dim)
            return output

        masks_list, categories = self._build_instance_masks(instance_mask, class_mask)
        num_instances = len(categories)

        # Build output tensors
        if self.max_objects is not None:
            mask_tensor = torch.zeros(self.max_objects, self.image_size, self.image_size, dtype=torch.float32)
            category_tensor = torch.full((self.max_objects,), -1, dtype=torch.long)
            for i, mask in enumerate(masks_list):
                if i >= self.max_objects:
                    break
                mask_tensor[i] = mask
                category_tensor[i] = categories[i]
        else:
            mask_tensor = torch.stack(masks_list) if masks_list else torch.zeros(0, self.image_size, self.image_size)
            category_tensor = torch.tensor(categories, dtype=torch.long) if categories else torch.empty(0, dtype=torch.long)

        output['masks'] = mask_tensor
        output['categories'] = category_tensor
        output['num_instances'] = torch.tensor(num_instances, dtype=torch.long)

        # Ignore mask (boundary pixels = 255)
        ignore_mask = torch.from_numpy((instance_mask == 255).astype(np.float32)).unsqueeze(0)
        output['ignore_mask'] = ignore_mask

        if self.return_properties:
            output['properties'] = self._build_properties(categories, masks_list)

        return output


class MOViDataset(Dataset):
    """
    MOVi (C or E) video dataset for object-centric learning.

    Returns a dictionary compatible with the training pipeline:
        - image: normalized image tensor (C, H, W)
        - masks: instance masks tensor (max_objects, H, W)
        - image_id: integer id tensor for the sample
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        *,
        max_objects: int = 25,
        image_size: int = 128,
        max_samples: Optional[int] = None,
        return_masks: bool = True,
        frames_per_clip: int = 24,
        predefined_json_paths: Optional[str] = None,
        horizontal_flip_prob: float = 0.0,
    ) -> None:
        """
        Args:
            data_root: Path to MOVi level directory (e.g., data/MOVi/c/train).
            split: 'train' or 'validation'.
            max_objects: Maximum number of objects to pad to.
            image_size: Target image size (square).
            max_samples: Maximum number of samples to load.
            return_masks: Whether to return masks (set False for training).
            frames_per_clip: Number of frames to sample per clip (for train).
            predefined_json_paths: Path to JSON file with pre-computed paths.
            horizontal_flip_prob: Probability of horizontal flip augmentation.
        """
        super().__init__()
        import glob
        from pathlib import Path

        self.data_root = data_root
        self.split = split
        self.max_objects = max_objects
        self.image_size = image_size
        self.return_masks = return_masks
        self.frames_per_clip = frames_per_clip
        self.horizontal_flip_prob = float(horizontal_flip_prob)

        # Get all video directories
        total_dirs = sorted(glob.glob(os.path.join(data_root, '*')))
        total_dirs = [d for d in total_dirs if os.path.isdir(d)]

        # Load or build path lists
        if split == 'train' and predefined_json_paths is not None and os.path.exists(predefined_json_paths):
            import json
            with open(predefined_json_paths, 'r') as fp:
                paths_persistence = json.load(fp)
            self.rgb = [Path(p) for p in paths_persistence['rgb']]
            self.mask = [[Path(p) for p in m] for m in paths_persistence['mask']]
        else:
            self.rgb = []
            self.mask = []
            import random

            for dir_path in total_dirs:
                image_paths = glob.glob(os.path.join(dir_path, '*_image.png'))
                if split == 'train':
                    random.shuffle(image_paths)
                    image_paths = image_paths[:frames_per_clip]
                else:
                    image_paths = sorted(image_paths)

                for image_path in image_paths:
                    p = Path(image_path)
                    self.rgb.append(p)
                    # Build mask paths for all possible objects
                    frame_id = p.stem.split('_')[0]
                    self.mask.append([
                        p.parent / f"{frame_id}_mask_{n:02}.png"
                        for n in range(max_objects)
                    ])

        if max_samples is not None:
            self.rgb = self.rgb[:max_samples]
            self.mask = self.mask[:max_samples]

        self.transform = make_transform(resize_size=image_size)

    def __len__(self) -> int:
        return len(self.rgb)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.rgb[idx]

        do_hflip = False
        if self.horizontal_flip_prob > 0.0:
            do_hflip = bool(torch.rand(1).item() < self.horizontal_flip_prob)

        img = Image.open(img_path).convert('RGB')
        if do_hflip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = self.transform(img)

        output: Dict[str, torch.Tensor] = {
            'image': img,
            'image_id': torch.tensor(idx),
        }

        if not self.return_masks:
            return output

        # Load instance masks
        mask_paths = self.mask[idx]
        masks_list = []
        for mask_path in mask_paths:
            if mask_path.exists():
                mask = Image.open(mask_path).convert('L')
                if do_hflip:
                    mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
                mask_tensor = torch.from_numpy(np.array(mask) > 127).float()
                masks_list.append(mask_tensor)
            else:
                masks_list.append(torch.zeros(self.image_size, self.image_size))

        mask_tensor = torch.stack(masks_list)

        # Count actual instances (masks with non-zero pixels)
        num_instances = sum(1 for m in masks_list if m.sum() > 0)

        output['masks'] = mask_tensor
        output['num_instances'] = torch.tensor(num_instances, dtype=torch.long)
        # MOVi has no semantic categories, use instance ID as category
        output['categories'] = torch.arange(self.max_objects, dtype=torch.long)
        output['ignore_mask'] = torch.zeros(1, self.image_size, self.image_size)

        return output


def get_voc_dataloaders(
    data_root: str,
    train_batch_size: int = 32,
    val_batch_size: Optional[int] = None,
    train_num_workers: int = 4,
    val_num_workers: Optional[int] = None,
    image_size: int = 256,
    max_objects: Optional[int] = 20,
    max_samples_train: Optional[int] = None,
    max_samples_val: Optional[int] = None,
    return_properties: bool = True,
    train_split: str = "trainaug",
    val_split: str = "val",
    train_return_masks: bool = True,
    val_return_masks: bool = True,
    train_horizontal_flip_prob: float = 0.5,
    val_horizontal_flip_prob: float = 0.0,
    train_pin_memory: bool = True,
    train_persistent_workers: Optional[bool] = None,
    train_prefetch_factor: Optional[int] = 2,
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create VOC dataloaders for train and validation.

    Args:
        data_root: Path to VOCdevkit/VOC2012 directory.
        train_batch_size: Batch size for the training dataloader.
        val_batch_size: Optional batch size for validation (defaults to train_batch_size).
        train_num_workers: Number of dataloader workers for training.
        val_num_workers: Optional number of workers for validation.
        image_size: Output image size (square).
        max_objects: Maximum number of objects allowed per image.
        max_samples_train: Optional limit on training samples.
        max_samples_val: Optional limit on validation samples.
        return_properties: Whether to return per-mask property vectors.
        train_split: Training split name ('trainaug' or 'train').
        val_split: Validation split name ('val').
        train_horizontal_flip_prob: Probability of horizontal flip for training.
        val_horizontal_flip_prob: Probability of horizontal flip for validation.
    """
    if val_batch_size is None:
        val_batch_size = train_batch_size
    if val_num_workers is None:
        val_num_workers = train_num_workers

    train_dataset = VOCDataset(
        data_root=data_root,
        split=train_split,
        max_objects=max_objects,
        image_size=image_size,
        max_samples=max_samples_train,
        return_masks=train_return_masks,
        return_properties=return_properties and train_return_masks,
        horizontal_flip_prob=train_horizontal_flip_prob,
    )

    val_dataset = VOCDataset(
        data_root=data_root,
        split=val_split,
        max_objects=max_objects,
        image_size=image_size,
        max_samples=max_samples_val,
        return_masks=val_return_masks,
        return_properties=return_properties and val_return_masks,
        horizontal_flip_prob=val_horizontal_flip_prob,
    )

    # persistent_workers requires num_workers > 0
    effective_persistent_workers = (
        (train_persistent_workers if train_persistent_workers is not None else True)
        and train_num_workers > 0
    )

    dataloaders = {
        'train': torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=train_num_workers,
            pin_memory=bool(train_pin_memory),
            persistent_workers=effective_persistent_workers,
            prefetch_factor=(
                train_prefetch_factor if train_num_workers > 0 else None
            ),
        ),
        'val': torch.utils.data.DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=val_num_workers,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=None,
        ),
    }

    return dataloaders


def get_movi_dataloaders(
    data_root: str,
    level: str = "c",
    train_batch_size: int = 32,
    val_batch_size: Optional[int] = None,
    train_num_workers: int = 4,
    val_num_workers: Optional[int] = None,
    image_size: int = 128,
    max_objects: int = 25,
    max_samples_train: Optional[int] = None,
    max_samples_val: Optional[int] = None,
    frames_per_clip: int = 24,
    train_return_masks: bool = True,
    val_return_masks: bool = True,
    train_horizontal_flip_prob: float = 0.5,
    val_horizontal_flip_prob: float = 0.0,
    train_pin_memory: bool = True,
    train_persistent_workers: Optional[bool] = None,
    train_prefetch_factor: Optional[int] = 2,
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create MOVi dataloaders for train and validation.

    Args:
        data_root: Path to MOVi directory (containing 'c' and/or 'e' subdirs).
        level: MOVi level ('c' or 'e').
        train_batch_size: Batch size for the training dataloader.
        val_batch_size: Optional batch size for validation.
        train_num_workers: Number of dataloader workers for training.
        val_num_workers: Optional number of workers for validation.
        image_size: Output image size (square).
        max_objects: Maximum number of objects (25 for MOVi).
        max_samples_train: Optional limit on training samples.
        max_samples_val: Optional limit on validation samples.
        frames_per_clip: Number of frames to sample per clip for training.
        train_horizontal_flip_prob: Probability of horizontal flip for training.
        val_horizontal_flip_prob: Probability of horizontal flip for validation.
    """
    if val_batch_size is None:
        val_batch_size = train_batch_size
    if val_num_workers is None:
        val_num_workers = train_num_workers

    train_root = os.path.join(data_root, level, 'train')
    val_root = os.path.join(data_root, level, 'validation')

    train_dataset = MOViDataset(
        data_root=train_root,
        split='train',
        max_objects=max_objects,
        image_size=image_size,
        max_samples=max_samples_train,
        return_masks=train_return_masks,
        frames_per_clip=frames_per_clip,
        horizontal_flip_prob=train_horizontal_flip_prob,
    )

    val_dataset = MOViDataset(
        data_root=val_root,
        split='validation',
        max_objects=max_objects,
        image_size=image_size,
        max_samples=max_samples_val,
        return_masks=val_return_masks,
        horizontal_flip_prob=val_horizontal_flip_prob,
    )

    # persistent_workers requires num_workers > 0
    effective_persistent_workers = (
        (train_persistent_workers if train_persistent_workers is not None else True)
        and train_num_workers > 0
    )

    dataloaders = {
        'train': torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=train_num_workers,
            pin_memory=bool(train_pin_memory),
            persistent_workers=effective_persistent_workers,
            prefetch_factor=(
                train_prefetch_factor if train_num_workers > 0 else None
            ),
        ),
        'val': torch.utils.data.DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=val_num_workers,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=None,
        ),
    }

    return dataloaders
