# To compute FID, first install pytorch_fid
# pip install pytorch-fid

import os
import cv2 as cv
from tqdm import tqdm
import shutil

from eval.score import *

cam_id = 18
ours_dir = './test_results/subject00/styleunet_gaussians3/testing__cam_%03d/batch_750000/rgb_map' % cam_id
posevocab_dir = './test_results/subject00/posevocab/testing__cam_%03d/rgb_map' % cam_id
tava_dir = './test_results/subject00/tava/cam_%03d' % cam_id
arah_dir = './test_results/subject00/arah/cam_%03d' % cam_id
slrf_dir = './test_results/subject00/slrf/cam_%03d' % cam_id
gt_dir = 'Z:/MultiviewRGB/THuman4/subject00/images/cam%02d' % cam_id
mask_dir = 'Z:/MultiviewRGB/THuman4/subject00/masks/cam%02d' % cam_id

frame_list = list(range(2000, 2500, 1))


if __name__ == '__main__':
    ours_metrics = Metrics()
    posevocab_metrics = Metrics()
    slrf_metrics = Metrics()
    arah_metrics = Metrics()
    tava_metrics = Metrics()

    shutil.rmtree('./tmp_quant')
    os.makedirs('./tmp_quant/ours', exist_ok = True)
    os.makedirs('./tmp_quant/posevocab', exist_ok = True)
    os.makedirs('./tmp_quant/slrf', exist_ok = True)
    os.makedirs('./tmp_quant/arah', exist_ok = True)
    os.makedirs('./tmp_quant/tava', exist_ok = True)
    os.makedirs('./tmp_quant/gt', exist_ok = True)

    for frame_id in tqdm(frame_list):
        ours_img = (cv.imread(ours_dir + '/%08d.jpg' % frame_id, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
        posevocab_img = (cv.imread(posevocab_dir + '/%08d.jpg' % frame_id, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
        slrf_img = (cv.imread(slrf_dir + '/%08d.png' % frame_id, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
        tava_img = (cv.imread(tava_dir + '/%d.jpg' % frame_id, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
        arah_img = (cv.imread(arah_dir + '/%d.jpg' % frame_id, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
        gt_img = (cv.imread(gt_dir + '/%08d.jpg' % frame_id, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
        mask_img = cv.imread(mask_dir + '/%08d.jpg' % frame_id, cv.IMREAD_UNCHANGED) > 128
        gt_img[~mask_img] = 1.

        ours_img_cropped, posevocab_img_cropped, slrf_img_cropped, tava_img_cropped, arah_img_cropped, gt_img_cropped = \
            crop_image(
                mask_img,
                512,
                ours_img,
                posevocab_img,
                slrf_img,
                tava_img,
                arah_img,
                gt_img
            )

        cv.imwrite('./tmp_quant/ours/%08d.png' % frame_id, (ours_img_cropped * 255).astype(np.uint8))
        cv.imwrite('./tmp_quant/posevocab/%08d.png' % frame_id, (posevocab_img_cropped * 255).astype(np.uint8))
        cv.imwrite('./tmp_quant/slrf/%08d.png' % frame_id, (slrf_img_cropped * 255).astype(np.uint8))
        cv.imwrite('./tmp_quant/tava/%08d.png' % frame_id, (tava_img_cropped * 255).astype(np.uint8))
        cv.imwrite('./tmp_quant/arah/%08d.png' % frame_id, (arah_img_cropped * 255).astype(np.uint8))
        cv.imwrite('./tmp_quant/gt/%08d.png' % frame_id, (gt_img_cropped * 255).astype(np.uint8))

        if ours_img is not None:
            ours_metrics.psnr += compute_psnr(ours_img, gt_img)
            ours_metrics.ssim += compute_ssim(ours_img, gt_img)
            ours_metrics.lpips += compute_lpips(ours_img_cropped, gt_img_cropped)
            ours_metrics.count += 1

        if posevocab_img is not None:
            posevocab_metrics.psnr += compute_psnr(posevocab_img, gt_img)
            posevocab_metrics.ssim += compute_ssim(posevocab_img, gt_img)
            posevocab_metrics.lpips += compute_lpips(posevocab_img_cropped, gt_img_cropped)
            posevocab_metrics.count += 1

        if slrf_img is not None:
            slrf_metrics.psnr += compute_psnr(slrf_img, gt_img)
            slrf_metrics.ssim += compute_ssim(slrf_img, gt_img)
            slrf_metrics.lpips += compute_lpips(slrf_img_cropped, gt_img_cropped)
            slrf_metrics.count += 1

        if arah_img is not None:
            arah_metrics.psnr += compute_psnr(arah_img, gt_img)
            arah_metrics.ssim += compute_ssim(arah_img, gt_img)
            arah_metrics.lpips += compute_lpips(arah_img_cropped, gt_img_cropped)
            arah_metrics.count += 1

        if tava_img is not None:
            tava_metrics.psnr += compute_psnr(tava_img, gt_img)
            tava_metrics.ssim += compute_ssim(tava_img, gt_img)
            tava_metrics.lpips += compute_lpips(tava_img_cropped, gt_img_cropped)
            tava_metrics.count += 1

    print('Ours metrics: ', ours_metrics)
    print('PoseVocab metrics: ', posevocab_metrics)
    print('SLRF metrics: ', slrf_metrics)
    print('ARAH metrics: ', arah_metrics)
    print('TAVA metrics: ', tava_metrics)

    print('--- Ours ---')
    os.system('python -m pytorch_fid --device cuda {} {}'.format('./tmp_quant/ours', './tmp_quant/gt'))
    print('--- PoseVocab ---')
    os.system('python -m pytorch_fid --device cuda {} {}'.format('./tmp_quant/posevocab', './tmp_quant/gt'))
    print('--- SLRF ---')
    os.system('python -m pytorch_fid --device cuda {} {}'.format('./tmp_quant/slrf', './tmp_quant/gt'))
    print('--- ARAH ---')
    os.system('python -m pytorch_fid --device cuda {} {}'.format('./tmp_quant/arah', './tmp_quant/gt'))
    print('--- TAVA ---')
    os.system('python -m pytorch_fid --device cuda {} {}'.format('./tmp_quant/tava', './tmp_quant/gt'))


