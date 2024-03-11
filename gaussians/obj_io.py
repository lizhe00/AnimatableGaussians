import torch
import numpy as np
from utils.sh_utils import RGB2SH, SH2RGB
from utils.general_utils import inverse_sigmoid
from plyfile import PlyData, PlyElement
import os


def construct_list_of_attributes(_features_dc, _features_rest, _scaling, _rotation):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(_features_dc.shape[1] * _features_dc.shape[2]):
        l.append('f_dc_{}'.format(i))
    for i in range(_features_rest.shape[1] * _features_rest.shape[2]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(_scaling.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(_rotation.shape[1]):
        l.append('rot_{}'.format(i))
    return l


def save_gaussians_as_ply(path, gaussian_vals: dict):
    os.makedirs(os.path.dirname(path), exist_ok = True)

    xyz = gaussian_vals['positions'].detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    fused_color = RGB2SH(gaussian_vals['colors'].detach()[:, [2, 1, 0]])
    features = torch.zeros((fused_color.shape[0], 3, (3 + 1) ** 2))
    features[:, :3, 0] = fused_color
    features_dc = features[:, :, 0:1].transpose(1, 2)
    features_rest = features[:, :, 1:].transpose(1, 2)
    f_dc = features_dc.transpose(1, 2).flatten(start_dim = 1).contiguous().cpu().numpy()
    f_rest = features_rest.transpose(1, 2).flatten(start_dim = 1).contiguous().cpu().numpy()
    opacities = inverse_sigmoid(gaussian_vals['opacity'].detach()).cpu().numpy()
    scale = torch.log(gaussian_vals['scales'].detach()).cpu().numpy()
    rotation = gaussian_vals['rotations'].detach().cpu().numpy()

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(features_dc, features_rest, scale, rotation)]

    elements = np.empty(xyz.shape[0], dtype = dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis = 1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)


def load_gaussians_from_ply(path):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])), axis = 1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    # assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (3 + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    # self._xyz = nn.Parameter(torch.tensor(xyz, dtype = torch.float, device = "cuda").requires_grad_(True))
    # self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype = torch.float, device = "cuda").transpose(1, 2).contiguous().requires_grad_(True))
    # self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype = torch.float, device = "cuda").transpose(1, 2).contiguous().requires_grad_(True))
    # self._opacity = nn.Parameter(torch.tensor(opacities, dtype = torch.float, device = "cuda").requires_grad_(True))
    # self._scaling = nn.Parameter(torch.tensor(scales, dtype = torch.float, device = "cuda").requires_grad_(True))
    # self._rotation = nn.Parameter(torch.tensor(rots, dtype = torch.float, device = "cuda").requires_grad_(True))
    #
    # self.active_sh_degree = self.max_sh_degree

    return {
        'positions': torch.tensor(xyz, dtype = torch.float, device = "cuda"),
        'colors': torch.tensor(SH2RGB(features_dc)[:, [2, 1, 0]], dtype = torch.float, device = "cuda").squeeze(-1),
        'opacity': torch.sigmoid(torch.tensor(opacities, dtype = torch.float, device = "cuda")),
        'scales': torch.exp(torch.tensor(scales, dtype = torch.float, device = "cuda")),
        'rotations': torch.nn.functional.normalize(torch.tensor(rots, dtype = torch.float, device = "cuda")),
        'features_extr': torch.tensor(features_extra, dtype = torch.float, device = "cuda")
    }
