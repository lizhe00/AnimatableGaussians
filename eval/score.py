import skimage.metrics
import numpy as np
import torch
import cv2 as cv

from network.lpips import LPIPS


class Metrics:
    def __init__(self):
        self.psnr = 0.
        self.ssim = 0.
        self.lpips = 0.
        self.count = 0

    def __repr__(self):
        if self.count > 0:
            return f'Count: {self.count}, PSNR: {self.psnr / self.count}, SSIM: {self.ssim / self.count}, LPIPS: {self.lpips / self.count}'
        else:
            return 'count is 0!'


def crop_image(gt_mask, patch_size, *args):
    """
    :param gt_mask: (H, W)
    :param patch_size: resize the cropped patch to the given patch_size
    :param args: some images with shape of (H, W, C)
    """
    mask_uv = np.argwhere(gt_mask > 0.)
    min_v, min_u = mask_uv.min(0)
    max_v, max_u = mask_uv.max(0)
    pad_size = 50
    min_v = (min_v - pad_size).clip(0, gt_mask.shape[0])
    min_u = (min_u - pad_size).clip(0, gt_mask.shape[1])
    max_v = (max_v + pad_size).clip(0, gt_mask.shape[0])
    max_u = (max_u + pad_size).clip(0, gt_mask.shape[1])
    len_v = max_v - min_v
    len_u = max_u - min_u
    max_size = max(len_v, len_u)

    cropped_images = []
    for image in args:
        if image is None:
            cropped_images.append(None)
        else:
            cropped_image = np.ones((max_size, max_size, 3), dtype = image.dtype)
            if len_v > len_u:
                start_u = (max_size - len_u) // 2
                cropped_image[:, start_u: start_u + len_u] = image[min_v: max_v, min_u: max_u]
            else:
                start_v = (max_size - len_v) // 2
                cropped_image[start_v: start_v + len_v, :] = image[min_v: max_v, min_u: max_u]

            cropped_image = cv.resize(cropped_image, (patch_size, patch_size), interpolation = cv.INTER_LINEAR)
            cropped_images.append(cropped_image)

    if len(cropped_images) > 1:
        return cropped_images
    else:
        return cropped_images[0]


def to_tensor(array, device = 'cuda'):
    if isinstance(array, np.ndarray):
        array = torch.from_numpy(array).to(device)
    elif isinstance(array, torch.Tensor):
        array = array.to(device)
    else:
        raise TypeError('Invalid type of array.')
    return array


def cut_rect(img):
    h, w = img.shape[:2]
    size = max(h, w)
    img_ = torch.ones((size, size, img.shape[2])).to(img)
    if h < w:
        img_[:h] = img
    else:
        img_[:, :w] = img
    return img_


lpips_net = None


def compute_lpips(src, tar, device = 'cuda'):
    src = to_tensor(src, device)
    tar = to_tensor(tar, device)
    global lpips_net
    if lpips_net is None:
        lpips_net = LPIPS(net = 'vgg').to(device)
    if src.shape[0] != src.shape[1]:
        src = cut_rect(src)
        tar = cut_rect(tar)
    with torch.no_grad():
        lpips = lpips_net.forward(src.permute(2, 0, 1)[None], tar.permute(2, 0, 1)[None], normalize = True).mean()
    return lpips.item()


def compute_psnr(src, tar):
    psnr = skimage.metrics.peak_signal_noise_ratio(tar, src, data_range=1)
    return psnr


def compute_ssim(src, tar):
    ssim = skimage.metrics.structural_similarity(src, tar, multichannel = True, data_range = 1)
    return ssim
