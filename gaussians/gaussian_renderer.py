#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization_depth_alpha import GaussianRasterizationSettings, GaussianRasterizer
from utils.graphics_utils import focal2fov, getProjectionMatrix
from utils.sh_utils import eval_sh


def render3(
    gaussian_vals: dict,
    bg_color: torch.Tensor,
    extr: torch.Tensor,
    intr: torch.Tensor,
    img_w: int,
    img_h: int,
    scaling_modifier = 1.0,
):
    means3D = gaussian_vals['positions']
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(means3D, dtype = means3D.dtype, requires_grad = True, device = "cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    means2D = screenspace_points
    opacity = gaussian_vals['opacity']

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    cov3D_precomp = None
    scales = gaussian_vals['scales']
    rotations = gaussian_vals['rotations']

    # Set up rasterization configuration
    FoVx = focal2fov(intr[0, 0].item(), img_w)
    FoVy = focal2fov(intr[1, 1].item(), img_h)
    tanfovx = math.tan(FoVx * 0.5)
    tanfovy = math.tan(FoVy * 0.5)
    world_view_transform = extr.transpose(1, 0).cuda()
    projection_matrix = getProjectionMatrix(znear = 0.1, zfar = 100, fovX = FoVx, fovY = FoVy, K = intr, img_w = img_w, img_h = img_h).transpose(0, 1).cuda()
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    camera_center = torch.linalg.inv(extr)[:3, 3]

    raster_settings = GaussianRasterizationSettings(
        image_height = img_h,
        image_width = img_w,
        tanfovx = tanfovx,
        tanfovy = tanfovy,
        bg = bg_color,
        scale_modifier = scaling_modifier,
        viewmatrix = world_view_transform,
        projmatrix = full_proj_transform,
        sh_degree = gaussian_vals['max_sh_degree'],
        campos = camera_center,
        prefiltered = False,
        debug = False
    )

    rasterizer = GaussianRasterizer(raster_settings = raster_settings)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    assert not ('colors' in gaussian_vals and 'shs' in gaussian_vals), "Cannot use both color and SH!"
    if 'colors' in gaussian_vals:
        colors_precomp = gaussian_vals['colors']
    else:
        colors_precomp = None
    if 'shs' in gaussian_vals:
        shs_view = gaussian_vals['shs']
        dir_pp = (means3D - camera_center.repeat(means3D.shape[0], 1))
        dir_pp_normalized = dir_pp / dir_pp.norm(dim = 1, keepdim = True)
        sh2rgb = eval_sh(gaussian_vals['max_sh_degree'], shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    shs = None

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "depth": rendered_depth,
        "mask": rendered_alpha,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii
    }

