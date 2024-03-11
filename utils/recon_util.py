import numpy as np
import skimage.measure as measure
import torch
import torch.nn.functional as F

import config


def extract_normal_volume(sdf_volume, voxel_size):
    sobel_x = torch.zeros((3, 3, 3), dtype = torch.float32, device = config.device)
    sobel_x[0] = torch.tensor([[-1,-2,-1], [-2,-4,-2], [-1,-2,-1]], dtype = torch.float32)
    sobel_x[2] = -sobel_x[0]
    sobel_z = sobel_x.permute((1, 2, 0))
    sobel_y = sobel_x.permute((2, 0, 1))

    # normalize
    sobel_x = sobel_x / (16 * 2 * voxel_size[0])
    sobel_y = sobel_y / (16 * 2 * voxel_size[1])
    sobel_z = sobel_z / (16 * 2 * voxel_size[2])

    sobel_filter = torch.cat((sobel_x.unsqueeze(0), sobel_y.unsqueeze(0), sobel_z.unsqueeze(0)), dim = 0)
    sobel_filter = sobel_filter.unsqueeze(1)

    normal_volume = F.conv3d(input = sdf_volume.view(1, 1, sdf_volume.shape[0], sdf_volume.shape[1], sdf_volume.shape[2]),
                             weight = sobel_filter,
                             padding = 1)
    normal_volume = normal_volume.squeeze()
    normal_volume = normal_volume.permute((1, 2, 3, 0))
    return normal_volume


def extract_normal_from_volume(sdf_volume, voxel_size, pts):
    '''
    :param sdf_volume:
    :param voxel_size: [vx, vy, vz]
    :param pts: (N, 3)
    :return:
    '''
    normal_volume = extract_normal_volume(sdf_volume, voxel_size)
    pts = pts[:, [2, 1, 0]]
    pts = pts.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    normals = F.grid_sample(normal_volume.permute((3, 0, 1, 2)).unsqueeze(0),
                            pts, padding_mode = 'border', align_corners = True)
    normals = normals.squeeze()
    normals = normals.permute((1, 0))
    normals_norm = torch.norm(normals, dim = 1, keepdim = True)
    normals /= normals_norm
    return normals


def recon_mesh(occ_volume, volume_res, bounds, volume_mask = None, iso_value = 0.5):
    """
    :param occ_volume: torch.Tensor
    :param volume_res: list or tuple
    :param bounds: numpy.ndarray (2, 3)
    :param iso_value: 0.5 for occupancy, 0 for sdf
    :return:
    """
    occ_volume = occ_volume.reshape(volume_res)
    if isinstance(volume_mask, torch.Tensor):
        volume_mask = volume_mask.cpu().numpy()
        volume_mask = volume_mask.reshape(volume_res)
    if isinstance(bounds, torch.Tensor):
        bounds = bounds.cpu().numpy()

    volume_len = bounds[1] - bounds[0]
    voxel_size = volume_len / np.array(volume_res, dtype = np.float32)

    vertices, faces, _, _ = measure.marching_cubes(occ_volume.cpu().numpy(), iso_value, spacing = voxel_size, mask = volume_mask)
    vertices = vertices + bounds[0] + 0.5 * voxel_size
    vertices_grid = 2 * (vertices - bounds[0]) / volume_len - 1.0
    normals = extract_normal_from_volume(occ_volume, voxel_size, torch.from_numpy(vertices_grid).to(occ_volume))
    normals = -normals.cpu().numpy()
    faces = faces[:, [2, 1, 0]]
    return vertices, faces, normals
