import math
import numpy as np
import torch
import torch.nn.functional as F
import cv2 as cv
import trimesh


def pos_map_to_mesh(pos_map: torch.Tensor):
    pd = (0, 0, 1 if pos_map.shape[1] % 2 == 1 else 0, 0, 1 if pos_map.shape[0] % 2 == 1 else 0, 0)
    pos_map = F.pad(pos_map, pd, 'constant', 0)
    mask = torch.linalg.norm(pos_map, dim = -1) > 0.1
    # cv.imshow('mask', mask.cpu().numpy().astype(np.uint8) * 255)
    mask = cv.erode(mask.cpu().numpy().astype(np.uint8), (5, 5), iterations = 20)
    mask = torch.from_numpy(mask > 0).to(pos_map.device)
    # cv.imshow('mask_eroded', mask.cpu().numpy().astype(np.uint8) * 255)
    # cv.waitKey(0)
    v0 = pos_map[:-1, :-1].reshape(-1, 3)
    v1 = pos_map[1:, :-1].reshape(-1, 3)
    v2 = pos_map[:-1, 1:].reshape(-1, 3)
    v3 = pos_map[1:, 1:].reshape(-1, 3)
    m0 = mask[:-1, :-1].reshape(-1)
    m1 = mask[1:, :-1].reshape(-1)
    m2 = mask[:-1, 1:].reshape(-1)
    m3 = mask[1:, 1:].reshape(-1)
    vertices = torch.cat([v0, v1, v2, v3], 0)
    masks = torch.cat([m0, m1, m2, m3], 0)
    pnum = v0.shape[0]

    a = torch.arange(0, pnum).to(torch.int64).to(pos_map.device)
    f1 = torch.stack([a, a + pnum, a + pnum * 2], 1)
    f2 = torch.stack([a + pnum, a + pnum * 3, a + pnum * 2], 1)
    faces = torch.cat([f1, f2], 0)

    # remove invalid faces
    face_mask = masks[faces.reshape(-1)].reshape(-1, 3).sum(1) == 3
    face_mask = torch.logical_and(face_mask, torch.linalg.norm(vertices[faces[:, 0]] - vertices[faces[:, 1]], dim = 1) < 0.02)
    face_mask = torch.logical_and(face_mask, torch.linalg.norm(vertices[faces[:, 1]] - vertices[faces[:, 2]], dim = 1) < 0.02)
    face_mask = torch.logical_and(face_mask, torch.linalg.norm(vertices[faces[:, 0]] - vertices[faces[:, 2]], dim = 1) < 0.02)
    valid_faces = faces[face_mask]

    # debug
    mesh = trimesh.Trimesh(vertices = vertices.cpu().numpy(), faces = valid_faces.cpu().numpy())
    mesh.export('./debug/mesh.obj')
    exit(1)


def to_HSV(c: torch.Tensor):
    """
    :param c: (N, 1) or (N,)
    :return: (N, 3)
    """
    h = (1 - c) * 240. / 60.
    x = 1 - torch.abs(h.to(torch.int64) % 2 + h - h.to(torch.int64) - 1.)

    rgb = torch.zeros((c.shape[0], 3)).to(c).to(torch.int64)

    cond_1 = torch.logical_and(h >= 0, h < 1)
    rgb[cond_1, 0] = 255
    rgb[cond_1, 1] = (x[cond_1] * 255).to(torch.int64)

    cond_2 = torch.logical_and(h >= 1, h < 2)
    rgb[cond_2, 0] = (x[cond_2] * 255).to(torch.int64)
    rgb[cond_2, 1] = 255

    cond_3 = torch.logical_and(h >= 2, h < 3)
    rgb[cond_3, 1] = 255
    rgb[cond_3, 2] = (x[cond_3] * 255).to(torch.int64)

    cond_4 = h >= 3
    rgb[cond_4, 1] = (x[cond_4] * 255).to(torch.int64)
    rgb[cond_4, 2] = 255

    rgb.clip_(0, 255)

    return rgb.to(torch.uint8)


# def calc_back_mv(dist):
#     rot_center = np.array([0, 0, dist], np.float32)
#     trans_mat = np.identity(4, np.float32)
#     trans_mat[:3, :3] = cv.Rodrigues(np.array([0, math.pi, 0]))[0]
#     trans_mat[:3, 3] = (np.identity(3) - trans_mat[:3, :3]) @ rot_center
#
#     return trans_mat


def calc_front_mv(object_center, tar_pos = np.array([0, 0, 2.0])):
    """
    calculate an extrinsic matrix for rendering the front of a 3D object
    under the assumption of fx,fy=550, cx,cy=256, img_h,img_w=512
    :param object_center: np.ndarray, (3,): the original center of the 3D object
    :param tar_pos: np.ndarray, (3,): the target center of the 3D object
    :return: extr_mat: np.ndarray, (4, 4)
    """
    mat_2origin = np.identity(4, np.float32)
    mat_2origin[:3, 3] = -object_center

    mat_rotX = np.identity(4, np.float32)
    mat_rotX[:3, :3] = cv.Rodrigues(np.array([math.pi, 0, 0]))[0]

    mat_2tarPos = np.identity(4, np.float32)
    mat_2tarPos[:3, 3] = tar_pos

    extr_mat = mat_2tarPos @ mat_rotX @ mat_2origin
    return extr_mat


def calc_back_mv(object_center, tar_pos = np.array([0, 0, 2.0])):
    """
    calculate an extrinsic matrix for rendering the back of a 3D object
    under the assumption of fx,fy=550, cx,cy=256, img_h,img_w=512
    :param object_center: np.ndarray, (3,): the original center of the 3D object
    :param tar_pos: np.ndarray, (3,): the target center of the 3D object
    :return: extr_mat: np.ndarray, (4, 4)
    """
    mat_2origin = np.identity(4, np.float32)
    mat_2origin[:3, 3] = -object_center

    mat_rotX = np.identity(4, np.float32)
    mat_rotX[:3, :3] = cv.Rodrigues(np.array([math.pi, 0, 0]))[0]

    mat_rotY = np.identity(4, np.float32)
    mat_rotY[:3, :3] = cv.Rodrigues(np.array([0, math.pi, 0]))[0]

    mat_2tarPos = np.identity(4, np.float32)
    mat_2tarPos[:3, 3] = tar_pos

    extr_mat = mat_2tarPos @ mat_rotY @ mat_rotX @ mat_2origin
    return extr_mat


def calc_free_mv(object_center, tar_pos = np.array([0, 0, 2.0]), rot_Y = 0., rot_X = 0., global_orient = None):
    """
    calculate an extrinsic matrix for rendering the back of a 3D object
    under the assumption of fx,fy=550, cx,cy=256, img_h,img_w=512
    :param object_center: np.ndarray, (3,): the original center of the 3D object
    :param tar_pos: np.ndarray, (3,): the target center of the 3D object
    :param rot_Y: float, rotation angle along Y axis
    :param global_orient: np.ndarray, global orientation of the 3D object
    :return: extr_mat: np.ndarray, (4, 4)
    """
    mat_2origin = np.identity(4, np.float32)
    mat_2origin[:3, 3] = -object_center

    mat_invGlobalOrient = np.identity(4, np.float32)
    if global_orient is not None:
        mat_invGlobalOrient[:3, :3] = cv.Rodrigues(np.array([math.pi, 0., 0.]))[0] @ np.linalg.inv(global_orient)
    else:
        mat_invGlobalOrient[:3, :3] = cv.Rodrigues(np.array([math.pi, 0., 0.]))[0]

    mat_rotY = np.identity(4, np.float32)
    mat_rotY[:3, :3] = cv.Rodrigues(np.array([0, rot_Y, 0]))[0]

    mat_rotX = np.identity(4, np.float32)
    mat_rotX[:3, :3] = cv.Rodrigues(np.array([rot_X, 0, 0]))[0]

    mat_2tarPos = np.identity(4, np.float32)
    mat_2tarPos[:3, 3] = tar_pos

    extr_mat = mat_2tarPos @ mat_rotX @ mat_rotY @ mat_invGlobalOrient @ mat_2origin
    return extr_mat


def calculate_cano_front_mv(mesh_center):
    if isinstance(mesh_center, torch.Tensor):
        mesh_center = mesh_center.cpu().numpy()
    front_mv = np.identity(4, np.float32)
    front_mv[:3, 3] = -mesh_center + np.array([0, 0, -10], np.float32)
    front_mv[1:3] *= -1
    return front_mv


def calculate_cano_back_mv(mesh_center):
    if isinstance(mesh_center, torch.Tensor):
        mesh_center = mesh_center.cpu().numpy()
    back_mv = np.identity(4, np.float32)
    rot_y = cv.Rodrigues(np.array([0, np.pi, 0], np.float32))[0]
    back_mv[:3, :3] = rot_y
    back_mv[:3, 3] = -rot_y @ mesh_center + np.array([0, 0, -10], np.float32)
    back_mv[1:3] *= -1
    return back_mv


def paper_visualize_pos_map(pos_map):
    mask = np.linalg.norm(pos_map, axis = -1) > 1e-6
    valid_pos = pos_map[mask]
    min_xyz = valid_pos.min(0)[None]
    max_xyz = valid_pos.max(0)[None]
    normalized_pos = (valid_pos - min_xyz) / (max_xyz - min_xyz)
    pos_map[mask] = normalized_pos
    pos_map[~mask] = np.array([0.5, 0.5, 0.5])
    return pos_map

def paper_visualize_gaussian_map(gaussian_map):
    mask = np.linalg.norm(gaussian_map, axis = -1) > 1e-6
    valid_gaussians = gaussian_map[mask]
    u, s, v = np.linalg.svd(valid_gaussians.transpose())
    print(u, s, v)
