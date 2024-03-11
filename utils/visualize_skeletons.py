import numpy as np
import torch
import trimesh
import cv2 as cv

import smplx
import config


def load_ball_cylinder():
    ball = trimesh.load(config.PROJ_DIR + '/assets/ball.obj', process = False)
    cylinder = trimesh.load(config.PROJ_DIR + '/assets/cylinder.obj', process = False)
    return ball, cylinder

ball, cylinder = load_ball_cylinder()


def construct_skeletons(joints, parent_ids):
    vertices = []
    faces = []
    vertex_num = 0
    for j in range(joints.shape[0]):
    # for j in [18, 20]:
        # add ball
        ball_v = np.array(ball.vertices).astype(np.float32)
        vertices.append(0.04 * ball_v + joints[j])
        faces.append(ball.faces + vertex_num)
        vertex_num += ball_v.shape[0]

        if parent_ids[j] >= 0:
            # add cylinder
            bone_len = np.linalg.norm(joints[j] - joints[parent_ids[j]])
            bone_d = 0.02
            cylinder_v = np.array(cylinder.vertices).astype(np.float32)
            cylinder_v[:, 1] = cylinder_v[:, 1] * bone_len / 1.0
            cylinder_v[:, [0, 2]] = cylinder_v[:, [0, 2]] * bone_d

            trans_j = np.identity(4, np.float32)
            trans_j[:3, 3] = joints[j] - np.array([0, -0.5 * bone_len, 0])
            d0 = np.array([0, 1, 0], np.float32)
            d1 = (joints[parent_ids[j]] - joints[j]) / bone_len
            cos_theta = np.dot(d0, d1)
            axis = np.cross(d0, d1)
            axis_norm = np.linalg.norm(axis)
            axis_angle = np.arccos(cos_theta) / axis_norm * axis
            rot = np.identity(4, np.float32)
            rot[:3, :3] = cv.Rodrigues(axis_angle)[0]
            rot[:3, 3] = -rot[:3, :3] @ joints[j] + joints[j]
            affine_mat = rot @ trans_j

            cylinder_v = np.einsum('ij,vj->vi', affine_mat[:3, :3], cylinder_v) + affine_mat[:3, 3]
            vertices.append(cylinder_v)
            faces.append(cylinder.faces + vertex_num)
            vertex_num += cylinder_v.shape[0]

    vertices = np.concatenate(vertices, 0)
    faces = np.concatenate(faces, 0)
    return vertices, faces


if __name__ == '__main__':
    smpl_params = np.load('G:/MultiviewRGB/subject00/smpl_params.npz')
    smpl_params = dict(smpl_params)
    smpl_params = {k: torch.from_numpy(v) for k, v in smpl_params.items()}

    smpl_model = smplx.SMPLX(model_path = config.PROJ_DIR + '/smpl_files/smplx', gender = 'neutral', use_pca = False, num_pca_comps = 45, flat_hand_mean = True, batch_size = 1)

    frame_id = 310
    # smpl_ret = smpl_model.forward(betas = smpl_params['betas'],
    #                               body_pose = smpl_params['body_pose'][frame_id].unsqueeze(0))
    smpl_ret = smpl_model.forward(betas = smpl_params['betas'],
                                  body_pose = config.cano_smpl_body_pose.unsqueeze(0))

    joints = smpl_ret.joints[0, :22].detach().cpu().numpy()
    parents = smpl_model.parents[:22]
    vertices, faces = construct_skeletons(joints, parents)

    skeleton_mesh = trimesh.Trimesh(vertices, faces, process = False)
    skeleton_mesh.export(config.PROJ_DIR + '/debug/skeleton_mesh.obj')
