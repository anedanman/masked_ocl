import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Tuple, Dict, List, Optional, Union, Any
import torchvision
from torchvision.transforms import v2


def make_transform(resize_size: int = 256):
    """Create image transform pipeline using torchvision v2.
    
    Args:
        resize_size: Target size for image resizing (square).
        
    Returns:
        Composed transform pipeline.
    """
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])


class CLEVRTEXDataset(Dataset):
    """
    CLEVRTEX Dataset for object-centric learning.
    
    Returns a dictionary in the same style as the COCO dataset implementation:
        - image: normalized image tensor (C, H, W)
        - masks: object masks tensor (max_objects, H, W)
        - image_id: integer id tensor for the sample
        - properties: one-hot encoded properties (max_objects, property_dim) [optional]
    """
    
    # Property vocabularies
    SIZES = ['small', 'medium', 'large']
    SHAPES = ['cube', 'cylinder', 'monkey', 'sphere']
    
    # Material vocabularies - all unique materials from CLEVRTEX
    # 'full' variant has 60 materials, 'outd' variant has 26 materials
    # Combined unique materials (86 total)
    MATERIALS = [
        'ambientcg_acoustic_foam_003',
        'ambientcg_fence007a',
        'ambientcg_marble016',
        'ambientcg_metal003',
        'ambientcg_tiles032',
        'ambientcg_tiles035',
        'cgbookcase_metalweave01_mr',
        'poly_haven_stony_dirt_path',
        'polyhaven_aerial_asphalt_01',
        'polyhaven_aerial_grass_rock',
        'polyhaven_aerial_mud_1',
        'polyhaven_aerial_rocks_01',
        'polyhaven_aerial_sand',
        'polyhaven_asphalt_02',
        'polyhaven_bark_willow',
        'polyhaven_blue_painted_planks',
        'polyhaven_book_pattern',
        'polyhaven_brick_floor',
        'polyhaven_brick_wall_005',
        'polyhaven_burned_ground_01',
        'polyhaven_ceramic_roof_01',
        'polyhaven_coast_sand_rocks_02',
        'polyhaven_concrete_floor',
        'polyhaven_concrete_floor_painted',
        'polyhaven_concrete_rock_path',
        'polyhaven_coral_fort_wall_02',
        'polyhaven_cracked_concrete_wall',
        'polyhaven_dark_wood',
        'polyhaven_denim_fabric',
        'polyhaven_dry_ground_rocks',
        'polyhaven_fabric_pattern_05',
        'polyhaven_fabric_pattern_07',
        'polyhaven_factory_brick',
        'polyhaven_factory_wall',
        'polyhaven_floor_pattern_02',
        'polyhaven_floor_tiles_06',
        'polyhaven_forest_floor',
        'polyhaven_forest_ground_04',
        'polyhaven_forrest_ground_01',
        'polyhaven_garage_floor',
        'polyhaven_green_metal_rust',
        'polyhaven_grey_plaster',
        'polyhaven_grey_roof_tiles',
        'polyhaven_hexagonal_concrete_paving',
        'polyhaven_kitchen_wood',
        'polyhaven_laminate_floor_02',
        'polyhaven_large_grey_tiles',
        'polyhaven_large_sandstone_blocks',
        'polyhaven_leather_red_02',
        'polyhaven_leaves_forest_ground',
        'polyhaven_medieval_blocks_02',
        'polyhaven_metal_plate',
        'polyhaven_mud_cracked_dry_03',
        'polyhaven_old_sandstone_02',
        'polyhaven_painted_metal_shutter',
        'polyhaven_plank_flooring_02',
        'polyhaven_preconcrete_wall_001',
        'polyhaven_raw_plank_wall',
        'polyhaven_red_brick_plaster_patch_02',
        'polyhaven_red_laterite_soil_stones',
        'polyhaven_red_sandstone_wall',
        'polyhaven_reed_roof_03',
        'polyhaven_rock_pitted_mossy',
        'polyhaven_rocky_gravel',
        'polyhaven_rocky_trail',
        'polyhaven_roof_07',
        'polyhaven_roots',
        'polyhaven_rough_plaster_brick_02',
        'polyhaven_rough_plaster_broken',
        'polyhaven_rough_wood',
        'polyhaven_rust_coarse_01',
        'polyhaven_rusty_metal',
        'polyhaven_rusty_metal_02',
        'polyhaven_slab_tiles',
        'polyhaven_snow_field_aerial',
        'polyhaven_square_floor_patern_01',
        'polyhaven_stone_wall',
        'polyhaven_weathered_brown_planks',
        'polyhaven_white_rough_plaster',
        'polyhaven_white_sandstone_blocks_02',
        'polyhaven_wood_floor_deck',
        'polyhaven_wood_peeling_paint_weathered',
        'polyhaven_wood_plank_wall',
        'polyhaven_wood_planks_grey',
        'sharetextures_chainmail_1',
        'whitemarble',
    ]
    
    def __init__(
        self,
        data_root: str,
        variant: str = 'full',  # 'full' or 'outd'
        max_objects: int = 10,
        image_size: int = 256,
        max_samples: Optional[int] = None,
        return_properties: bool = True,
        return_masks: bool = True,
        samples: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Args:
            data_root: Path to clevrtexv2 directory
            variant: 'full' or 'outd'
            max_objects: Maximum number of objects to pad to
            image_size: Target image size (square, will be resized to image_size x image_size)
            max_samples: Maximum number of samples to load (None for all)
            return_properties: Whether to return property vectors
        """
        super().__init__()
        self.data_root = data_root
        self.variant = variant
        self.max_objects = max_objects
        self.image_size = image_size
        self.return_properties = return_properties
        self.return_masks = return_masks
        if self.return_properties and not self.return_masks:
            raise ValueError("Cannot return properties when masks are disabled.")
        
        # Create transform pipeline
        self.transform = make_transform(resize_size=image_size)
        
        # Calculate property dimension
        # [size(3), shape(4), material(86), presence(1)]
        self.property_dim = len(self.SIZES) + len(self.SHAPES) + len(self.MATERIALS) + 1
        
        # Load all scene metadata
        if samples is not None:
            self.samples = samples[: max_samples] if max_samples is not None else list(samples)
        else:
            self.samples = []
            base_path = os.path.join(data_root, f'clevrtexv2_{variant}')
            
            # Scan all folders
            for folder in sorted(os.listdir(base_path)):
                folder_path = os.path.join(base_path, folder)
                if not os.path.isdir(folder_path):
                    continue
                    
                for scene_dir in sorted(os.listdir(folder_path)):
                    scene_path = os.path.join(folder_path, scene_dir)
                    if not os.path.isdir(scene_path):
                        continue
                    
                    json_path = os.path.join(scene_path, f'{scene_dir}.json')
                    if os.path.exists(json_path):
                        self.samples.append({
                            'scene_dir': scene_path,
                            'scene_name': scene_dir,
                            'json_path': json_path
                        })
                        
                        if max_samples and len(self.samples) >= max_samples:
                            break
                
                if max_samples and len(self.samples) >= max_samples:
                    break
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _load_image(self, scene_dir: str, filename_base: str) -> torch.Tensor:
        """Load and transform image using torchvision v2."""
        # CLEVRTEX images have format: {filename_base}0003.png
        img_path = os.path.join(scene_dir, f'{filename_base}0003.png')
        img = Image.open(img_path).convert('RGB')
        
        # Apply transforms (resize, convert to tensor, normalize)
        img_tensor = self.transform(img)
        
        return img_tensor
    
    def _load_masks(self, scene_dir: str, mask_filename_base: str, num_objects: int) -> torch.Tensor:
        """Load segmentation masks."""
        # CLEVRTEX masks have format: {mask_filename_base}_0003.png
        mask_path = os.path.join(scene_dir, f'{mask_filename_base}_0003.png')
        mask_img = Image.open(mask_path)
        
        # Resize to square if needed
        if mask_img.size != (self.image_size, self.image_size):
            mask_img = mask_img.resize((self.image_size, self.image_size), Image.NEAREST)
        
        mask_array = np.array(mask_img)
        
        # Create one mask per object
        masks = torch.zeros(self.max_objects, self.image_size, self.image_size)
        
        for i in range(num_objects):
            masks[i] = torch.from_numpy(mask_array == (i)).float()
    
        return masks
    
    def _encode_properties(self, objects: List[Dict]) -> torch.Tensor:
        """
        Encode object properties as one-hot vectors with presence indicator.
        
        Property vector structure:
        [size_onehot(3), shape_onehot(4), material_onehot(86), presence(1)]
        Total: 94 dimensions
        """
        properties = torch.zeros(self.max_objects, self.property_dim)
        
        for i, obj in enumerate(objects[:self.max_objects]):
            offset = 0
            
            # Size (3 dimensions)
            size_idx = self.SIZES.index(obj['size'])
            properties[i, offset + size_idx] = 1.0
            offset += len(self.SIZES)
            
            # Shape (4 dimensions)
            shape_idx = self.SHAPES.index(obj['shape'])
            properties[i, offset + shape_idx] = 1.0
            offset += len(self.SHAPES)
            
            # Material (86 dimensions)
            material_idx = self.MATERIALS.index(obj['material'])
            properties[i, offset + material_idx] = 1.0
            offset += len(self.MATERIALS)
            
            # Presence (1 dimension)
            properties[i, offset] = 1.0
        
        return properties
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict containing:
                - image: (C, H, W) tensor
                - masks: (max_objects, H, W) or (num_instances, H, W) binary masks
                - categories: tensor of category ids (padded with -1 when necessary)
                - num_instances: number of valid masks before padding
                - areas: normalized areas for each mask
                - bboxes: normalized bounding boxes [x, y, w, h]
                - properties: optional property vectors when ``return_properties`` is True
        """
        """
        Returns a dict with keys similar to COCO samples:
            - image: (C, H, W) normalized image
            - masks: (max_objects, H, W) object masks
            - image_id: tensor([idx]) for consistency
            - properties: (max_objects, property_dim) one-hot properties when enabled
        """
        sample = self.samples[idx]
        
        # Load scene metadata
        with open(sample['json_path'], 'r') as f:
            scene = json.load(f)
        
        # Load image
        image = self._load_image(sample['scene_dir'], scene['image_filename'])
        
        output: Dict[str, torch.Tensor] = {
            'image': image,
            'image_id': torch.tensor(idx)
        }
        
        if self.return_masks:
            masks = self._load_masks(sample['scene_dir'], scene['mask_filename'], scene['num_objects'])
            output['masks'] = masks
        
        if self.return_properties:
            properties = self._encode_properties(scene['objects'])
            output['properties'] = properties
        
        return output


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
        horizontal_flip_prob: float = 0.0,
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
        if self.return_properties and not self.return_masks:
            raise ValueError("Cannot return properties when masks are disabled.")
        self.horizontal_flip_prob = float(horizontal_flip_prob)
        if not 0.0 <= self.horizontal_flip_prob <= 1.0:
            raise ValueError(
                f"horizontal_flip_prob must be between 0 and 1 (got {horizontal_flip_prob})."
            )

        self.transform = make_transform(resize_size=image_size)

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
        self.coco = COCO(ann_file)
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

        self.filtered_out = 0

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, sample: Dict[str, Any], horizontal_flip: bool = False) -> torch.Tensor:
        img_path = os.path.join(self.image_dir, sample["image_file"])
        img = Image.open(img_path).convert("RGB")
        if horizontal_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return self.transform(img)

    def _resize_mask(self, mask: np.ndarray) -> torch.Tensor:
        mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
        mask_img = mask_img.resize((self.image_size, self.image_size), Image.NEAREST)
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
        metadata: List[Dict[str, Any]],
        original_size: Tuple[int, int],
    ) -> torch.Tensor:
        num_instances = len(categories)
        rows = self.max_objects if self.max_objects is not None else num_instances
        properties = torch.zeros(rows, self.property_dim, dtype=torch.float32)

        orig_h, orig_w = original_size
        for idx, cat_id in enumerate(categories):
            if cat_id in self.category_id_to_idx:
                cat_idx = self.category_id_to_idx[cat_id]
                properties[idx, cat_idx] = 1.0

            mask_bool = metadata[idx]["mask"]
            ys, xs = np.nonzero(mask_bool)
            if len(xs) == 0:
                continue

            xmin, xmax = xs.min(), xs.max()
            ymin, ymax = ys.min(), ys.max()

            center_x = ((xmin + xmax + 1) / 2.0) / max(orig_w, 1)
            center_y = ((ymin + ymax + 1) / 2.0) / max(orig_h, 1)

            properties[idx, self.num_categories] = float(2 * center_x - 1)
            properties[idx, self.num_categories + 1] = float(2 * center_y - 1)
            properties[idx, self.num_categories + 2] = 1.0

        return properties

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
            props = self._build_properties(
                category_ids, metadata, (sample["height"], sample["width"])
            )
            output["properties"] = props

        # Areas and bounding boxes (normalized to [0, 1])
        if self.max_objects is not None:
            areas_tensor = torch.zeros(self.max_objects, dtype=torch.float32)
            bboxes_tensor = torch.zeros(self.max_objects, 4, dtype=torch.float32)
        else:
            areas_tensor = torch.zeros(num_instances, dtype=torch.float32)
            bboxes_tensor = torch.zeros(num_instances, 4, dtype=torch.float32)

        orig_area = max(sample["width"] * sample["height"], 1)
        for i in range(min(num_instances, len(metadata))):
            meta = metadata[i]
            area_px = float(meta.get("area", 0.0))
            areas_tensor[i] = area_px / orig_area

            x, y, bw, bh = self._bbox_from_mask(
                meta["mask"], sample["width"], sample["height"]
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
    train_horizontal_flip_prob: float = 0.5,
    val_horizontal_flip_prob: float = 0.0,
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

    train_dataset = COCODataset(
        data_root=data_root,
        split=train_split,
        mode=mode,
        max_objects=max_objects,
        image_size=image_size,
        max_samples=max_samples_train,
        min_area=min_area,
        return_properties=return_properties and train_return_masks,
        return_masks=train_return_masks,
        horizontal_flip_prob=train_horizontal_flip_prob,
    )

    val_dataset = COCODataset(
        data_root=data_root,
        split=val_split,
        mode=mode,
        max_objects=max_objects,
        image_size=image_size,
        max_samples=max_samples_val,
        min_area=min_area,
        return_properties=return_properties and val_return_masks,
        return_masks=val_return_masks,
        horizontal_flip_prob=val_horizontal_flip_prob,
    )

    dataloaders = {
        "train": torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=train_num_workers,
            pin_memory=True,
            persistent_workers=True if train_num_workers > 0 else False,
            prefetch_factor=2 if train_num_workers > 0 else None,
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


def get_clevrtex_dataloaders(
    data_root: str,
    variant: str = 'full',
    train_batch_size: int = 32,
    val_batch_size: Optional[int] = None,
    test_batch_size: Optional[int] = None,
    train_num_workers: int = 4,
    val_num_workers: Optional[int] = None,
    test_num_workers: Optional[int] = None,
    image_size: int = 256,
    max_objects: int = 10,
    max_samples: Optional[int] = None,
    return_properties: bool = True,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    train_return_masks: bool = True,
    val_return_masks: bool = True,
    test_return_masks: bool = True,
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create CLEVRTEX dataloaders with train/val/test splits.
    
    Args:
        data_root: Path to clevrtexv2 directory
        variant: 'full' or 'outd'
        train_batch_size: Training batch size
        val_batch_size: Optional validation batch size (defaults to training size)
        test_batch_size: Optional test batch size (defaults to validation size)
        train_num_workers: Number of workers for the training loader
        val_num_workers: Optional number of workers for validation (defaults to training)
        test_num_workers: Optional number of workers for test (defaults to validation)
        image_size: Target image size (square)
        max_objects: Maximum number of objects
        max_samples: Maximum number of samples to load (None for all)
        return_properties: Whether to return property vectors
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        test_ratio: Ratio of data for testing
        seed: Random seed for reproducible splitting
        
    Returns:
        Dictionary of dataloaders with keys 'train', 'val', 'test'
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "train_ratio + val_ratio + test_ratio must equal 1.0"
    
    # Create full dataset to determine splits
    full_dataset = CLEVRTEXDataset(
        data_root=data_root,
        variant=variant,
        max_objects=max_objects,
        image_size=image_size,
        max_samples=max_samples,
        return_properties=return_properties,
        return_masks=True,
    )
    
    # Get total size and compute split sizes
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    # Create reproducible random split
    rng = np.random.RandomState(seed)
    indices = rng.permutation(total_size)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_samples = [full_dataset.samples[i] for i in train_indices]
    val_samples = [full_dataset.samples[i] for i in val_indices]
    test_samples = [full_dataset.samples[i] for i in test_indices]

    def build_dataset(sample_list: List[Dict[str, Any]], return_masks: bool) -> CLEVRTEXDataset:
        return CLEVRTEXDataset(
            data_root=data_root,
            variant=variant,
            max_objects=max_objects,
            image_size=image_size,
            max_samples=None,
            return_properties=return_properties and return_masks,
            return_masks=return_masks,
            samples=sample_list,
        )

    if val_batch_size is None:
        val_batch_size = train_batch_size
    if test_batch_size is None:
        test_batch_size = val_batch_size
    if val_num_workers is None:
        val_num_workers = train_num_workers
    if test_num_workers is None:
        test_num_workers = val_num_workers

    train_dataset = build_dataset(train_samples, train_return_masks)
    val_dataset = build_dataset(val_samples, val_return_masks)
    test_dataset = build_dataset(test_samples, test_return_masks)
    
    # Create dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=train_num_workers,
            pin_memory=True,
            persistent_workers=True if train_num_workers > 0 else False,
            prefetch_factor=2 if train_num_workers > 0 else None,
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
        'test': torch.utils.data.DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=test_num_workers,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=None,
        )
    }
    
    return dataloaders
