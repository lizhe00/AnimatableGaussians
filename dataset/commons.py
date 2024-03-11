import numpy as np
import torch
import trimesh

import config


def _initialize_hands(self):
    smplx_lhand_to_mano_rhand_data = np.load(config.PROJ_DIR + '/smpl_files/mano/smplx_lhand_to_mano_rhand.npz', allow_pickle = True)
    smplx_rhand_to_mano_rhand_data = np.load(config.PROJ_DIR + '/smpl_files/mano/smplx_rhand_to_mano_rhand.npz', allow_pickle = True)
    smpl_lhand_vert_id = np.copy(smplx_lhand_to_mano_rhand_data['smpl_vert_id_to_mano'])
    smpl_rhand_vert_id = np.copy(smplx_rhand_to_mano_rhand_data['smpl_vert_id_to_mano'])
    self.smpl_lhand_vert_id = torch.from_numpy(smpl_lhand_vert_id)
    self.smpl_rhand_vert_id = torch.from_numpy(smpl_rhand_vert_id)
    self.smpl_hands_vert_id = torch.cat([self.smpl_lhand_vert_id, self.smpl_rhand_vert_id], 0)
    mano_face_closed = np.loadtxt(config.PROJ_DIR + '/smpl_files/mano/mano_face_close.txt').astype(np.int64)
    self.mano_face_closed = torch.from_numpy(mano_face_closed)
    self.mano_face_closed_turned = self.mano_face_closed[:, [2, 1, 0]]
    self.mano_face_closed_2hand = torch.cat([self.mano_face_closed[:, [2, 1, 0]], self.mano_face_closed + self.smpl_lhand_vert_id.shape[0]], 0)


def generate_two_manos(self, smplx_verts: torch.Tensor):
    left_mano_v = smplx_verts[self.smpl_lhand_vert_id].cpu().numpy()
    left_mano_trimesh = trimesh.Trimesh(left_mano_v, self.mano_face_closed_turned, process = False)
    left_mano_n = left_mano_trimesh.vertex_normals.astype(np.float32)

    right_mano_v = smplx_verts[self.smpl_rhand_vert_id].cpu().numpy()
    right_mano_trimesh = trimesh.Trimesh(right_mano_v, self.mano_face_closed, process = False)
    right_mano_n = right_mano_trimesh.vertex_normals.astype(np.float32)

    return left_mano_v, left_mano_n, right_mano_v, right_mano_n
