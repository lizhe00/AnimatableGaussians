import os
import numpy as np
import torch
import torch.nn.functional as F

import config


def compute_gradient_volume(weight_volume, voxel_size):
    """
    :param weight_volume: (C, X, Y, Z)
    """
    sobel_x = torch.zeros((3, 3, 3), dtype = torch.float32, device = config.device)
    sobel_x[0] = torch.tensor([[-1,-2,-1], [-2,-4,-2], [-1,-2,-1]], dtype = torch.float32)
    sobel_x[2] = -sobel_x[0]
    sobel_z = sobel_x.permute((1, 2, 0))
    sobel_y = sobel_x.permute((2, 0, 1))

    # normalize
    sobel_x = sobel_x / (16 * 2 * voxel_size[0])
    sobel_y = sobel_y / (16 * 2 * voxel_size[1])
    sobel_z = sobel_z / (16 * 2 * voxel_size[2])

    # sobel_x = torch.zeros((3, 3, 3), dtype = torch.float32, device = config.device)
    # sobel_x[0] = torch.tensor([[0, 0, 0], [0, -1, 0], [0, 0, 0]], dtype = torch.float32)
    # sobel_x[2] = -sobel_x[0]
    # sobel_z = sobel_x.permute((1, 2, 0))
    # sobel_y = sobel_x.permute((2, 0, 1))
    #
    # # normalize
    # sobel_x = sobel_x / (2 * voxel_size[0])
    # sobel_y = sobel_y / (2 * voxel_size[1])
    # sobel_z = sobel_z / (2 * voxel_size[2])

    sobel_filter = torch.cat((sobel_x.unsqueeze(0), sobel_y.unsqueeze(0), sobel_z.unsqueeze(0)), dim = 0)
    sobel_filter = sobel_filter.unsqueeze(1)

    grad_volume = F.conv3d(input = weight_volume.unsqueeze(1), weight = sobel_filter, padding = 1)
    return grad_volume  # [J, 3, X, Y, Z]


class CanoBlendWeightVolume:
    def __init__(self, data_path):
        if not os.path.exists(data_path):
            raise FileNotFoundError('# CanoBlendWeightVolume is not found from %s' % data_path)
        data = np.load(data_path)

        diff_weight_volume = data['diff_weight_volume']
        diff_weight_volume = diff_weight_volume.transpose((3, 0, 1, 2))[None]
        # base_weight_volume = base_weight_volume.transpose((3, 2, 1, 0))[None]
        self.diff_weight_volume = torch.from_numpy(diff_weight_volume).to(torch.float32).to(config.device)
        self.res_x, self.res_y, self.res_z = self.diff_weight_volume.shape[2:]
        self.joint_num = self.diff_weight_volume.shape[1]

        self.ori_weight_volume = torch.from_numpy(data['ori_weight_volume'].transpose((3, 0, 1, 2))[None]).to(torch.float32).to(config.device)

        if 'sdf_volume' in data:
            smpl_sdf_volume = data['sdf_volume']
            if len(smpl_sdf_volume.shape) == 3:
                smpl_sdf_volume = smpl_sdf_volume[..., None]
            smpl_sdf_volume = smpl_sdf_volume.transpose((3, 0, 1, 2))[None]
            self.smpl_sdf_volume = torch.from_numpy(smpl_sdf_volume).to(torch.float32).to(config.device)

        self.volume_bounds = torch.from_numpy(data['volume_bounds']).to(torch.float32).to(config.device)
        self.center = torch.from_numpy(data['center']).to(torch.float32).to(config.device)
        self.smpl_bounds = torch.from_numpy(data['smpl_bounds']).to(torch.float32).to(config.device)

        volume_len = self.volume_bounds[1] - self.volume_bounds[0]
        self.voxel_size = volume_len / torch.tensor([self.res_x-1, self.res_y-1, self.res_z-1]).to(volume_len)
        # self.base_gradient_volume = compute_gradient_volume(self.diff_weight_volume[0], self.voxel_size)  # [joint_num, 3, X, Y, Z]

    def forward_weight(self, pts, requires_scale = True, volume_type = 'diff'):
        """
        :param pts: (B, N, 3)
        :param requires_scale: bool, scale pts to [0, 1]
        :return: (B, N, 24)
        """
        if requires_scale:
            pts = (pts - self.volume_bounds[None, None, 0]) / (self.volume_bounds[1] - self.volume_bounds[0])[None, None]
        B, N, _ = pts.shape
        grid = 2 * pts - 1
        grid = grid[..., [2, 1, 0]]
        grid = grid[:, :, None, None]

        weight_volume = self.diff_weight_volume if volume_type == 'diff' else self.ori_weight_volume

        base_w = F.grid_sample(weight_volume.expand(B, -1, -1, -1, -1),
                               grid,
                               mode = 'bilinear',
                               padding_mode = 'border',
                               align_corners = True)
        base_w = base_w[:, :, :, 0, 0].permute(0, 2, 1)
        return base_w

    def forward_weight_grad(self, pts, requires_scale = True):
        """
        :param pts: (B, N, 3)
        :param requires_scale: bool, scale pts to [0, 1]
        :return: (B, N, 24)
        """
        if requires_scale:
            pts = (pts - self.volume_bounds[None, None, 0]) / (self.volume_bounds[1] - self.volume_bounds[0])[None, None]
        B, N, _ = pts.shape
        grid = 2 * pts - 1
        grid = grid.reshape(-1, 3)[:, [2, 1, 0]]
        grid = grid[None, :, None, None]

        base_g = F.grid_sample(self.base_gradient_volume.view(self.joint_num * 3, self.res_x, self.res_y, self.res_z)[None].expand(B, -1, -1, -1, -1),
                               grid,
                               mode = 'nearest',
                               padding_mode = 'border',
                               align_corners = True)
        base_g = base_g[:, :, :, 0, 0].permute(0, 2, 1).reshape(B, N, -1, 3)
        return base_g

    def forward_sdf(self, pts, requires_scale = True):
        if requires_scale:
            pts = (pts - self.volume_bounds[None, None, 0]) / (self.volume_bounds[1] - self.volume_bounds[0])[None, None]
        B, N, _ = pts.shape
        grid = 2 * pts - 1
        grid = grid.reshape(-1, 3)[:, [2, 1, 0]]
        grid = grid[None, :, None, None]

        sdf = F.grid_sample(self.smpl_sdf_volume.expand(B, -1, -1, -1, -1),
                            grid,
                            padding_mode = 'border',
                            align_corners = True)
        sdf = sdf[:, :, :, 0, 0].permute(0, 2, 1)

        return sdf
