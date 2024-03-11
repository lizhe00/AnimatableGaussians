import torch
import torch.nn.functional as F
import numpy as np
import cv2


def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [4, 4]
    """
    xyz = np.dot(xyz, RT[:3, :3].T) + RT[:3, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy


def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d


def get_bound_2d_mask(bounds, K, pose, H, W):
    corners_3d = get_bound_corners(bounds)
    corners_2d = project(corners_3d, K, pose)
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 4]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask


def get_near_far(bounds, ray_o, ray_d):
    """calculate intersections with 3d bounding box"""
    bounds = bounds + np.array([-0.01, 0.01])[:, None]
    nominator = bounds[None] - ray_o[:, None]
    # calculate the step of intersections at six planes of the 3d bounding box
    d_intersect = (nominator / (ray_d[:, None] + 1e-9)).reshape(-1, 6)
    # calculate the six interections
    p_intersect = d_intersect[..., None] * ray_d[:, None] + ray_o[:, None]
    # calculate the intersections located at the 3d bounding box
    min_x, min_y, min_z, max_x, max_y, max_z = bounds.ravel()
    eps = 1e-6
    p_mask_at_box = (p_intersect[..., 0] >= (min_x - eps)) * \
                    (p_intersect[..., 0] <= (max_x + eps)) * \
                    (p_intersect[..., 1] >= (min_y - eps)) * \
                    (p_intersect[..., 1] <= (max_y + eps)) * \
                    (p_intersect[..., 2] >= (min_z - eps)) * \
                    (p_intersect[..., 2] <= (max_z + eps))
    # obtain the intersections of rays which intersect exactly twice
    mask_at_box = p_mask_at_box.sum(-1) == 2
    p_intervals = p_intersect[mask_at_box][p_mask_at_box[mask_at_box]].reshape(
        -1, 2, 3)

    # calculate the step of intersections
    ray_o = ray_o[mask_at_box]
    ray_d = ray_d[mask_at_box]
    norm_ray = np.linalg.norm(ray_d, axis=1)
    d0 = np.linalg.norm(p_intervals[:, 0] - ray_o, axis=1) / norm_ray
    d1 = np.linalg.norm(p_intervals[:, 1] - ray_o, axis=1) / norm_ray
    near = np.minimum(d0, d1)
    far = np.maximum(d0, d1)

    return near, far, mask_at_box


def get_rays(uv, extr, intr):
    inv_extr = np.linalg.inv(extr)
    cam_loc = inv_extr[:3, 3]

    num_samples, _ = uv.shape

    depth = np.ones((num_samples, 1)).astype(uv.dtype)
    pixel_2d = np.concatenate([uv, depth], -1)

    inv_intr = np.linalg.inv(intr)
    pixel_points_cam = np.einsum('ij,nj->ni', inv_intr, pixel_2d)

    world_coords = np.einsum('ij,nj->ni', inv_extr[:3, :3], pixel_points_cam) + inv_extr[:3, 3]
    ray_dirs = world_coords - cam_loc[None]
    ray_dirs /= (np.linalg.norm(ray_dirs, axis = -1, keepdims = True) + 1e-8)

    return ray_dirs, cam_loc[None].repeat(num_samples, axis = 0)


def sample_pts_on_rays(ray_o, ray_d, near, far, N_samples = 64, perturb = False, depth_guided_mask = None):
    # calculate the steps for each ray
    t_vals = torch.linspace(0., 1., steps = N_samples).to(near)
    z_vals = near[..., None] * (1. - t_vals) + far[..., None] * t_vals

    if perturb:
        # only perturb for depth_guided_mask == True
        # get intervals between samples
        if depth_guided_mask is None:
            depth_guided_mask = torch.ones(ray_o.shape[:-1], dtype = torch.bool, device = ray_o.device)
        # mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        # upper = torch.cat([mids, z_vals[..., -1:]], -1)
        # lower = torch.cat([z_vals[..., :1], mids], -1)
        # # stratified samples in those intervals
        # t_rand = torch.rand(z_vals.shape).to(upper)
        # z_vals = lower + (upper - lower) * t_rand

        z_vals = z_vals.view(-1, N_samples)
        depth_guided_mask = depth_guided_mask.view(-1)
        mids = .5 * (z_vals[depth_guided_mask, 1:] + z_vals[depth_guided_mask, :-1])
        upper = torch.cat([mids, z_vals[depth_guided_mask, -1:]], -1)
        lower = torch.cat([z_vals[depth_guided_mask, :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals[depth_guided_mask].shape).to(upper)
        z_vals[depth_guided_mask] = lower + (upper - lower) * t_rand
        z_vals = z_vals.view(list(ray_o.shape[:-1]) + [N_samples])

    pts = ray_o[:, :, None] + ray_d[:, :, None] * z_vals[..., None]

    return pts, z_vals


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples).to(weights)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples]).to(weights)

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u).to(weights)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    # inds = searchsorted(cdf, u, side='right')
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def sample_pts_on_rays_fine(rays_o, rays_d, z_vals, weights, N_importance, perturb=0.):
    batch_size, rays_per_batch = rays_o.shape[:2]

    z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    z_samples = sample_pdf(
        z_vals_mid.view(batch_size*rays_per_batch, -1),
        weights[..., 1:-1].view(batch_size*rays_per_batch, -1),
        N_importance, det=(perturb==0.), pytest=False)
    z_samples = z_samples.detach().view(batch_size, rays_per_batch, -1)

    z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]
    return pts, z_vals


def raw2outputs(raw, z_vals, white_bkgd = False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    rgb = raw[..., :-1]  # [N_rays, N_samples, 3]
    alpha = raw[..., -1]

    weights = alpha * torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1. - alpha + 1e-10], -1), -1)[..., :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map).to(depth_map),
                              depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def raw2outputs2(rgb, alpha, z_vals, white_bkgd = False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """

    weights = alpha * torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1. - alpha + 1e-10], -1), -1)[..., :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map).to(depth_map),
                              depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def gen_uv(img_w, img_h):
    x, y = np.meshgrid(np.linspace(0, img_w - 1, img_w, dtype = np.int),
                       np.linspace(0, img_h - 1, img_h, dtype = np.int))
    uv = np.stack([x, y], axis = -1)
    return uv


def sample_randomly_for_nerf_rendering(color_img,
                                       mask_img,
                                       depth_img,
                                       extr,
                                       intr,
                                       live_bounds,
                                       sample_num = 1024,
                                       inside_radio = 0.5,
                                       unsample_region_mask = None):
    assert color_img.shape[:2] == mask_img.shape[:2] and color_img.shape[:2] == depth_img.shape[:2]
    assert 0. <= inside_radio <= 1.0
    img_h, img_w = color_img.shape[:2]
    bound_mask = get_bound_2d_mask(live_bounds, intr, extr, img_h, img_w) > 0
    # cv2.imshow('bound_mask_0', bound_mask.astype(np.uint8) * 255)
    # cv2.waitKey(0)
    if unsample_region_mask is not None:
        bound_mask = np.logical_and(bound_mask, unsample_region_mask < 1e-6)
    # cv2.imshow('bound_mask_1', bound_mask.astype(np.uint8) * 255)
    # cv2.waitKey(0)
    uv_img = gen_uv(img_w, img_h)
    inside_uv = uv_img[np.logical_and(mask_img, bound_mask > 1e-6)]
    outside_uv = uv_img[np.logical_and(~mask_img, bound_mask > 1e-6)]
    count = 0
    uv, ray_o, ray_d, near, far = [], [], [], [], []
    while count < sample_num:
        rest_num = sample_num - count
        inside_sample_num = int(rest_num * inside_radio)
        outside_sample_num = rest_num - inside_sample_num
        sampled_inside_uv = inside_uv[np.random.choice(inside_uv.shape[0], inside_sample_num, replace = False)]
        sampled_outside_uv = outside_uv[np.random.choice(outside_uv.shape[0], outside_sample_num, replace = False)]
        uv_ = np.concatenate([sampled_inside_uv, sampled_outside_uv], axis = 0)
        ray_d_, ray_o_ = get_rays(uv_, extr, intr)
        near_, far_, mask_at_bound = get_near_far(live_bounds, ray_o_, ray_d_)
        uv.append(uv_[mask_at_bound])
        ray_o.append(ray_o_[mask_at_bound])
        ray_d.append(ray_d_[mask_at_bound])
        near.append(near_)
        far.append(far_)
        count += near_.shape[0]
    uv = np.concatenate(uv, 0)
    ray_o = np.concatenate(ray_o, 0)
    ray_d = np.concatenate(ray_d, 0)
    near = np.concatenate(near, 0)
    far = np.concatenate(far, 0)

    # gt
    color_gt = color_img[uv[:, 1], uv[:, 0]]
    mask_gt = mask_img[uv[:, 1], uv[:, 0]]
    depth_gt = depth_img[uv[:, 1], uv[:, 0]]
    color_gt[mask_gt < 1e-6] = 0

    # distance to depth if depth is available
    x = (uv[:, 0] + 0.5 - intr[0, 2]) * depth_gt / intr[0, 0]
    y = (uv[:, 1] + 0.5 - intr[1, 2]) * depth_gt / intr[1, 1]
    dist = np.sqrt(x * x + y * y + depth_gt * depth_gt).astype(np.float32)

    ret = {
        'uv': uv,
        'ray_o': ray_o,
        'ray_d': ray_d,
        'near': near,
        'far': far,
        'color_gt': color_gt,
        'mask_gt': mask_gt.astype(np.float32),
        'depth_gt': depth_gt,
        'dist': dist
    }

    return ret


def sample_patch_for_nerf_rendering(color_img,
                                    mask_img,
                                    depth_img,
                                    extr,
                                    intr,
                                    live_bounds,
                                    patch_num = 2,
                                    patch_size = 32,
                                    inside_radio = 0.5,
                                    unsample_region_mask = None,
                                    resize_factor = 1.0):
    assert color_img.shape[:2] == mask_img.shape[:2] and color_img.shape[:2] == depth_img.shape[:2]
    assert 0. <= inside_radio <= 1.0
    if resize_factor != 1.0:
        color_img = cv2.resize(color_img, (0, 0), fx = resize_factor, fy = resize_factor, interpolation = cv2.INTER_NEAREST)
        mask_img = cv2.resize(mask_img.astype(np.uint8), (0, 0), fx = resize_factor, fy = resize_factor, interpolation = cv2.INTER_NEAREST) > 0
        depth_img = cv2.resize(depth_img, (0, 0), fx = resize_factor, fy = resize_factor, interpolation = cv2.INTER_NEAREST)
        if unsample_region_mask is not None:
            unsample_region_mask = cv2.resize(unsample_region_mask.astype(np.uint8), (0, 0), fx = resize_factor, fy = resize_factor, interpolation = cv2.INTER_NEAREST) > 0
        intr_ = intr.copy()
        intr_[:2] *= resize_factor
        intr = intr_

    img_h, img_w = color_img.shape[:2]
    bound_mask = get_bound_2d_mask(live_bounds, intr, extr, img_h, img_w) > 0.
    # cv2.imshow('bound_mask', bound_mask.astype(np.uint8) * 255)
    # cv2.waitKey(0)
    if unsample_region_mask is not None:
        bound_mask = np.logical_and(bound_mask, unsample_region_mask < 1e-6)
    # cv2.imshow('unsample_region', unsample_region_mask.astype(np.uint8) * 255)
    # cv2.imshow('bound_mask', bound_mask.astype(np.uint8) * 255)
    # cv2.waitKey(0)
    uv_img = gen_uv(img_w, img_h)
    inside_uv = uv_img[np.logical_and(mask_img, bound_mask > 1e-6)]
    outside_uv = uv_img[np.logical_and(~mask_img, bound_mask > 1e-6)]
    # cv2.imwrite('./debug/mask_img.png', mask_img.astype(np.uint8) * 255)
    # cv2.imwrite('./debug/bound_mask.png', bound_mask.astype(np.uint8) * 255)
    # exit(1)
    # cv2.imshow('mask_img', mask_img.astype(np.uint8) * 255)
    # cv2.imshow('bound_mask', bound_mask.astype(np.uint8) * 255)
    # cv2.waitKey(0)
    uv, ray_o, ray_d, near, far, mask_within_patch = [], [], [], [], [], []
    for patch_idx in range(patch_num):
        p = np.random.rand()
        if p < inside_radio:
            candidate_uv = inside_uv
        else:
            candidate_uv = outside_uv
        center_uv = candidate_uv[np.random.choice(candidate_uv.shape[0], 1, replace = False)[0]]
        half_patch_size = patch_size // 2
        u_min = np.clip(center_uv[0] - half_patch_size, 0, img_w - patch_size)
        v_min = np.clip(center_uv[1] - half_patch_size, 0, img_h - patch_size)
        u_max = u_min + patch_size
        v_max = v_min + patch_size

        uv_patch = uv_img[v_min: v_max, u_min: u_max].reshape(-1, 2)
        bound_mask_patch = bound_mask[v_min: v_max, u_min: u_max].reshape(-1)
        ray_d_patch, ray_o_patch = get_rays(uv_patch[bound_mask_patch], extr, intr)
        near_patch, far_patch, mask_at_bound = get_near_far(live_bounds, ray_o_patch, ray_d_patch)
        ray_o.append(ray_o_patch[mask_at_bound])
        ray_d.append(ray_d_patch[mask_at_bound])
        near.append(near_patch)
        far.append(far_patch)
        uv.append(uv_patch)
        bound_mask_patch_pos = np.argwhere(bound_mask_patch)
        bound_mask_patch[bound_mask_patch_pos[~mask_at_bound]] = False
        mask_within_patch.append(bound_mask_patch)

    uv = np.concatenate(uv, 0)
    mask_within_patch = np.concatenate(mask_within_patch, 0)
    ray_o = np.concatenate(ray_o, 0)
    ray_d = np.concatenate(ray_d, 0)
    near = np.concatenate(near, 0)
    far = np.concatenate(far, 0)

    # gt
    color_gt = color_img[uv[:, 1], uv[:, 0]]
    mask_gt = mask_img[uv[:, 1], uv[:, 0]]
    depth_gt = depth_img[uv[:, 1], uv[:, 0]]
    color_gt[mask_gt < 1e-6] = 0

    # distance to depth if depth is available
    x = (uv[:, 0] + 0.5 - intr[0, 2]) * depth_gt / intr[0, 0]
    y = (uv[:, 1] + 0.5 - intr[1, 2]) * depth_gt / intr[1, 1]
    dist = np.sqrt(x * x + y * y + depth_gt * depth_gt).astype(np.float32)[mask_within_patch]

    ret = {
        'uv': uv,
        'mask_within_patch': mask_within_patch,
        'ray_o': ray_o,
        'ray_d': ray_d,
        'near': near,
        'far': far,
        'color_gt': color_gt,
        'mask_gt': mask_gt.astype(np.float32),
        'depth_gt': depth_gt,
        'dist': dist
    }

    return ret


def sample_randomly_for_nerf_rendering_wSideViewMask(
        color_img,
        mask_img,
        depth_img,
        extr,
        intr,
        live_bounds,
        sample_num = 1024,
        inside_radio = 0.5,
        side_view_radio = 0.8,
        unsample_region_mask = None,
        side_view_mask = None
):
    assert color_img.shape[:2] == mask_img.shape[:2] and color_img.shape[:2] == depth_img.shape[:2]
    assert 0. <= inside_radio <= 1.0
    img_h, img_w = color_img.shape[:2]
    bound_mask = get_bound_2d_mask(live_bounds, intr, extr, img_h, img_w) > 0
    # cv2.imshow('bound_mask', bound_mask.astype(np.uint8) * 255)
    # cv2.waitKey(0)
    if unsample_region_mask is not None:
        bound_mask = np.logical_and(bound_mask, unsample_region_mask < 1e-6)
    uv_img = gen_uv(img_w, img_h)
    inside_uv_sideView = uv_img[np.logical_and(np.logical_and(mask_img, bound_mask), side_view_mask)]
    inside_uv_fbView = uv_img[np.logical_and(np.logical_and(mask_img, bound_mask), ~side_view_mask)]
    outside_uv = uv_img[np.logical_and(~mask_img, bound_mask)]
    count = 0
    uv, ray_o, ray_d, near, far = [], [], [], [], []
    while count < sample_num:
        rest_num = sample_num - count
        inside_sample_num_1 = int(rest_num * inside_radio * side_view_radio)
        inside_sample_num_2 = int(rest_num * inside_radio * (1. - side_view_radio))
        outside_sample_num = rest_num - (inside_sample_num_1 + inside_sample_num_2)
        sampled_inside_uv_1 = inside_uv_sideView[np.random.choice(inside_uv_sideView.shape[0], inside_sample_num_1, replace = inside_sample_num_1 < inside_uv_sideView.shape[0])]
        sampled_inside_uv_2 = inside_uv_fbView[np.random.choice(inside_uv_fbView.shape[0], inside_sample_num_2, replace = inside_sample_num_2 < inside_uv_fbView.shape[0])]
        sampled_outside_uv = outside_uv[np.random.choice(outside_uv.shape[0], outside_sample_num, replace = False)]
        uv_ = np.concatenate([sampled_inside_uv_1, sampled_inside_uv_2, sampled_outside_uv], axis = 0)
        ray_d_, ray_o_ = get_rays(uv_, extr, intr)
        near_, far_, mask_at_bound = get_near_far(live_bounds, ray_o_, ray_d_)
        uv.append(uv_[mask_at_bound])
        ray_o.append(ray_o_[mask_at_bound])
        ray_d.append(ray_d_[mask_at_bound])
        near.append(near_)
        far.append(far_)
        count += near_.shape[0]
    uv = np.concatenate(uv, 0)
    ray_o = np.concatenate(ray_o, 0)
    ray_d = np.concatenate(ray_d, 0)
    near = np.concatenate(near, 0)
    far = np.concatenate(far, 0)

    # gt
    color_gt = color_img[uv[:, 1], uv[:, 0]]
    mask_gt = mask_img[uv[:, 1], uv[:, 0]]
    depth_gt = depth_img[uv[:, 1], uv[:, 0]]
    color_gt[mask_gt < 1e-6] = 0

    # distance to depth if depth is available
    x = (uv[:, 0] + 0.5 - intr[0, 2]) * depth_gt / intr[0, 0]
    y = (uv[:, 1] + 0.5 - intr[1, 2]) * depth_gt / intr[1, 1]
    dist = np.sqrt(x * x + y * y + depth_gt * depth_gt).astype(np.float32)

    ret = {
        'uv': uv,
        'ray_o': ray_o,
        'ray_d': ray_d,
        'near': near,
        'far': far,
        'color_gt': color_gt,
        'mask_gt': mask_gt.astype(np.float32),
        'depth_gt': depth_gt,
        'dist': dist
    }

    # color_img[uv[:, 1], uv[:, 0]] = 255
    # cv2.imshow('color', color_img)
    # cv2.waitKey(0)

    return ret

