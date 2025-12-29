import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as VF
from scipy import ndimage

from src.utils import denormalize_image

MAX_ITER = 10
POS_W = 7 
POS_XY_STD = 3
Bi_W = 10
Bi_XY_STD = 50 
Bi_RGB_STD = 5



def dense_crf(image, mask):
    h, w = mask.shape
    mask = mask.reshape(1, h, w)
    fg = mask.astype(np.float) 
    bg = 1 - fg
    output_logits = torch.from_numpy(np.concatenate((bg,fg), axis=0))

    #image = np.array(VF.to_pil_image(image_tensor))[:, :, ::-1]
    H, W = image.shape[:2]
    image = np.ascontiguousarray(image)
    
    output_logits = F.interpolate(output_logits.unsqueeze(0), size=(H, W), mode="bilinear",
                                  align_corners=False).squeeze()
    output_probs = F.softmax(output_logits, dim=0).cpu().numpy()

    c = output_probs.shape[0]
    h = output_probs.shape[1]
    w = output_probs.shape[2]

    U = utils.unary_from_softmax(output_probs)
    U = np.ascontiguousarray(U)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=image, compat=Bi_W)

    Q = d.inference(MAX_ITER)
    Q = np.array(Q).reshape((c, h, w))
    MAP = np.argmax(Q, axis=0).reshape((h,w)).astype(np.float32)
    return MAP


def crf_refine(image, masks, denorm_img=True):
    """
    Refine the masks using dense CRF.
    Args:
        image: (3, H, W), torch tensor
        masks: (C, H, W), torch tensor, where C is the number of masks. Masks are probs in [0, 1].
        denorm_img: whether to denormalize the image. Default is True.
    """
    
    # denorm img and convert to np.uint8
    if denorm_img:
        image = denormalize_image(image)
    image = (image * 255).byte().permute(1, 2, 0).cpu().numpy()
    image = np.ascontiguousarray(image)

    C, H, W = masks.shape
    d = dcrf.DenseCRF2D(W, H, C)
    U = utils.unary_from_softmax(masks.cpu().numpy())
    U = np.ascontiguousarray(U)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=image, compat=Bi_W)
    Q = d.inference(MAX_ITER)
    Q = np.array(Q).reshape((C, H, W))
    refined_masks = torch.from_numpy(Q).float()
    return refined_masks

def crf_refine_batch(images, masks, denorm_img=True):
    """
    Refine the masks using dense CRF.
    Args:
        images: (B, 3, H, W), torch tensor
        masks: (B, C, H, W), torch tensor, where C is the number of masks. Masks are probs in [0, 1].
        denorm_img: whether to denormalize the image. Default is True.
    """
    B, C, H, W = masks.shape
    refined_masks = []
    for i in range(B):
        refined_mask = crf_refine(images[i], masks[i], denorm_img)
        refined_masks.append(refined_mask)
    refined_masks = torch.stack(refined_masks, dim=0)
    return refined_masks