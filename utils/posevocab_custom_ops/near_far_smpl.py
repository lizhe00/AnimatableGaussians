from __future__ import division, print_function
import os
import torch
import numpy as np
from torch.utils.cpp_extension import load

# file_dir = os.path.dirname(os.path.realpath(__file__))
# sources = [file_dir + '/near_far_smpl.cpp', file_dir + '/near_far_smpl_kernel.cu']
#
# near_far_smpl_bridge = load(
#     name='near_far_smpl_bridge',
#     sources=sources,
#     # extra_include_paths=['/usr/include/python2.7'],
#     verbose=True)
import posevocab_custom_ops


def near_far_smpl(vertices, ray_o, ray_d, radius = 0.1):
    vertices = vertices.contiguous().to(torch.float32)
    ray_o = ray_o.contiguous().to(torch.float32)
    ray_d = ray_d.contiguous().to(torch.float32)
    ray_num = ray_o.shape[0]
    near = torch.cuda.FloatTensor(ray_num).fill_(0.0).contiguous()
    far = torch.cuda.FloatTensor(ray_num).fill_(0.0).contiguous()
    intersect_flag = torch.cuda.BoolTensor(ray_num).fill_(0.0).contiguous()
    posevocab_custom_ops.near_far_smpl(vertices, ray_o, ray_d, near, far, intersect_flag, radius)
    return near, far, intersect_flag



