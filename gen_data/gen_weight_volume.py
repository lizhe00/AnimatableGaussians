import os
import glob
import numpy as np
import igl
import pytorch3d.ops
import tqdm

import smplx
import torch
import trimesh
import config


tmp_dir = './debug/tmp'
os.makedirs(tmp_dir, exist_ok = True)
depth = 7


def compute_lbs_grad(cano_smpl: trimesh.Trimesh, vertex_lbs: np.ndarray):
    vertices = cano_smpl.vertices.astype(np.float32)
    normals = cano_smpl.vertex_normals.copy().astype(np.float32)
    faces = cano_smpl.faces.astype(np.int64)

    normals /= np.linalg.norm(normals, axis = 1, keepdims = True)
    tx = np.cross(normals, np.array([[0, 0, 1]], np.float32))
    tx /= np.linalg.norm(tx, axis = 1, keepdims = True)
    ty = np.cross(normals, tx)

    vnum = vertices.shape[0]
    fnum = faces.shape[0]
    jnum = vertex_lbs.shape[1]
    lbs_grad_tx = np.zeros((vnum, jnum), np.float32)
    lbs_grad_ty = np.zeros((vnum, jnum), np.float32)
    for vidx in tqdm.tqdm(range(vnum)):
        v = vertices[vidx]
        for jidx in range(jnum):
            for neighbor_vidx in cano_smpl.vertex_neighbors[vidx]:
                vn = vertices[neighbor_vidx]
                pos_diff = vn - v
                pos_diff_norm = np.linalg.norm(pos_diff)
                val_diff = vertex_lbs[neighbor_vidx, jidx] - vertex_lbs[vidx, jidx]
                val_diff /= pos_diff_norm
                pos_diff /= pos_diff_norm

                lbs_grad_tx[vidx, jidx] += val_diff * np.dot(pos_diff, tx[vidx])
                lbs_grad_ty[vidx, jidx] += val_diff * np.dot(pos_diff, ty[vidx])

            lbs_grad_tx[vidx, jidx] /= float(len(cano_smpl.vertex_neighbors[vidx]))
            lbs_grad_ty[vidx, jidx] /= float(len(cano_smpl.vertex_neighbors[vidx]))

    # print(lbs_grad_ty)
    # print(lbs_grad_tx)
    lbs_grad = lbs_grad_tx[:, :, None] * tx[:, None] + lbs_grad_ty[:, :, None] * ty[:, None]  # (V, J, 3)
    for jid in tqdm.tqdm(range(jnum)):
        out_fn_grad = os.path.join(tmp_dir, f"cano_data_lbs_grad_{jid:02d}.xyz")
        out_fn_val = os.path.join(tmp_dir, f"cano_data_lbs_val_{jid:02d}.xyz")

        out_data_grad = np.concatenate([vertices, lbs_grad[:, jid]], 1)
        out_data_val = np.concatenate([vertices, vertex_lbs[:, jid:jid+1]], 1)
        np.savetxt(out_fn_grad, out_data_grad, fmt="%.8f")
        np.savetxt(out_fn_val, out_data_val, fmt="%.8f")


def solve(num_joints, point_interpolant_exe):
    for jid in range(num_joints):
        print('Solving joint %d' % jid)
        cmd = f'{point_interpolant_exe} ' + \
            f'--inValues {os.path.join(tmp_dir, f"cano_data_lbs_val_{jid:02d}.xyz")} ' + \
            f'--inGradients {os.path.join(tmp_dir, f"cano_data_lbs_grad_{jid:02d}.xyz")} ' + \
            f'--gradientWeight 0.05 --dim 3 --verbose ' + \
            f'--grid {os.path.join(tmp_dir, f"grid_{jid:02d}.grd")} ' + \
            f'--depth {depth} '

        os.system(cmd)


@torch.no_grad()
def calc_cano_weight_volume(data_dir, gender = 'neutral'):
    smpl_params = np.load(data_dir + '/smpl_params.npz')
    smpl_shape = torch.from_numpy(smpl_params['betas'][0]).to(torch.float32)
    smpl_model = smplx.SMPLX(model_path = config.PROJ_DIR + '/smpl_files/smplx', gender = gender, use_pca = False, num_pca_comps = 45, flat_hand_mean = True, batch_size = 1)

    def get_grid_points(bounds, res):
        # voxel_size = (bounds[1] - bounds[0]) / (np.array(res, np.float32) - 1)
        # x = np.arange(bounds[0, 0], bounds[1, 0] + voxel_size[0], voxel_size[0])
        # y = np.arange(bounds[0, 1], bounds[1, 1] + voxel_size[1], voxel_size[1])
        # z = np.arange(bounds[0, 2], bounds[1, 2] + voxel_size[2], voxel_size[2])
        x = np.linspace(bounds[0, 0], bounds[1, 0], res[0])
        y = np.linspace(bounds[0, 1], bounds[1, 1], res[1])
        z = np.linspace(bounds[0, 2], bounds[1, 2], res[2])

        pts = np.stack(np.meshgrid(x, y, z, indexing = 'ij'), axis = -1)
        return pts

    if isinstance(smpl_model, smplx.SMPLX):
        cano_smpl = smpl_model.forward(betas = smpl_shape[None],
                                       global_orient = config.cano_smpl_global_orient[None],
                                       transl = config.cano_smpl_transl[None],
                                       body_pose = config.cano_smpl_body_pose[None])
    elif isinstance(smpl_model, smplx.SMPL):
        cano_smpl = smpl_model.forward(betas = smpl_shape[None],
                                       global_orient = config.cano_smpl_global_orient[None],
                                       transl = config.cano_smpl_transl[None],
                                       body_pose = config.cano_smpl_pose[6:][None])
    else:
        raise TypeError('Not support this SMPL type.')

    cano_smpl_trimesh = trimesh.Trimesh(
        cano_smpl.vertices[0].cpu().numpy(),
        smpl_model.faces,
        process = False
    )

    compute_lbs_grad(cano_smpl_trimesh, smpl_model.lbs_weights.cpu().numpy())
    solve(smpl_model.lbs_weights.shape[-1], ".\\bins\\PointInterpolant.exe")

    ### NOTE concatenate all grids
    fn_list = sorted(list(glob.glob(os.path.join(tmp_dir, 'grid_*.grd'))))

    grids = []
    import array
    for fn in fn_list:
        with open(fn, 'rb') as f:
            bytes = f.read()
        grid_res = 2 ** depth
        grid_header_len = len(bytes) - grid_res ** 3 * 8
        grid_np = np.array(array.array('d', bytes[grid_header_len:])).reshape(grid_res, grid_res, grid_res)
        grids.append(grid_np)

    grids_all = np.stack(grids, 0)
    grids_all = np.clip(grids_all, 0.0, 1.0)
    grids_all = grids_all / grids_all.sum(0)[None]
    # print(grids_all.shape)
    # np.save(join(data_templates_path, subject, subject + '_cano_lbs_weights_grid_float32.npy'), grids_all.astype(np.float32))
    diff_weights = grids_all.transpose((3, 2, 1, 0))  # convert to xyz
    min_xyz = cano_smpl_trimesh.vertices.min(0).astype(np.float32)
    max_xyz = cano_smpl_trimesh.vertices.max(0).astype(np.float32)
    max_len = 1.1 * (max_xyz - min_xyz).max()
    center = 0.5 * (min_xyz + max_xyz)
    volume_bounds = np.stack(
        [center - 0.5 * max_len, center + 0.5 * max_len], 0
    )

    min_xyz[:2] -= 0.05
    max_xyz[:2] += 0.05
    min_xyz[2] -= 0.15
    max_xyz[2] += 0.15
    smpl_bounds = np.stack(
        [min_xyz, max_xyz], 0
    )

    res = diff_weights.shape[:3]
    pts = get_grid_points(volume_bounds, res)
    pts = pts.reshape(-1, 3)
    dists, face_id, closest_pts = igl.signed_distance(pts, cano_smpl_trimesh.vertices, smpl_model.faces.astype(np.int32))
    triangles = cano_smpl_trimesh.vertices[smpl_model.faces[face_id]]
    weights = smpl_model.lbs_weights.numpy()[smpl_model.faces[face_id]]
    barycentric_weight = trimesh.triangles.points_to_barycentric(triangles, closest_pts)
    ori_weights = (barycentric_weight[:, :, None] * weights).sum(1)
    # weights[dists > 0.08] = 0.
    dists = dists.reshape(res).astype(np.float32)
    ori_weights = ori_weights.reshape(list(res) + [-1]).astype(np.float32)

    np.savez(data_dir + '/cano_weight_volume.npz',
             diff_weight_volume = diff_weights.astype(np.float32),
             ori_weight_volume = ori_weights.astype(np.float32),
             sdf_volume = -dists,
             volume_bounds = volume_bounds,
             smpl_bounds = smpl_bounds,
             center = center)

    # # debug
    # from network.volume import CanoBlendWeightVolume
    # from utils.smpl_util import skinning
    # weight_volume = CanoBlendWeightVolume(data_dir + '/cano_weight_volume.npz')
    # pts_w = weight_volume.forward_weight(torch.from_numpy(cano_smpl_trimesh.vertices).to(torch.float32).to(config.device))
    # pts_w = pts_w[0].cpu().numpy()
    # np.savetxt('../debug/pts_w_query.txt', pts_w, fmt = '%.8f')
    # np.savetxt('../debug/pts_w_val.txt', smpl_model.lbs_weights.cpu().numpy(), fmt = '%.8f')
    #
    # # smpl_data = np.load(data_dir + '/smpl_params.npz')
    # smpl_data = np.load('F:/pose/thuman4/pose_01.npz')
    # smpl_data = {k: torch.from_numpy(v).to(torch.float32) for k, v in smpl_data.items()}
    # frame_idx = 61
    # posed_smpl = smpl_model.forward(
    #     betas = smpl_data['betas'],
    #     global_orient = smpl_data['global_orient'][frame_idx: frame_idx+1],
    #     transl = smpl_data['transl'][frame_idx: frame_idx+1],
    #     body_pose = smpl_data['body_pose'][frame_idx: frame_idx+1]
    # )
    # jnt_mats = torch.matmul(posed_smpl.A, cano_smpl.A.inverse())
    # pts_w = torch.from_numpy(pts_w).to(torch.float32)
    # cano_verts = torch.from_numpy(cano_smpl_trimesh.vertices).to(torch.float32)
    # posed_v_query = skinning(cano_verts[None], pts_w[None], jnt_mats)
    # posed_v_ori = skinning(cano_verts[None], smpl_model.lbs_weights[None], jnt_mats)
    # posed_smpl_query = trimesh.Trimesh(posed_v_query.cpu().numpy()[0], smpl_model.faces, process = False)
    # posed_smpl_ori = trimesh.Trimesh(posed_v_ori.cpu().numpy()[0], smpl_model.faces, process = False)
    # posed_smpl_query.export('../debug/posed_smpl_query.obj')
    # posed_smpl_ori.export('../debug/posed_smpl_ori.obj')
    # cano_smpl_trimesh.export('../debug/cano_smpl.obj')
    #
    # cano_mesh = trimesh.load('../debug/cano_mesh.ply', process = False)
    # cano_mesh_v = torch.from_numpy(cano_mesh.vertices).to(torch.float32)
    # cano_mesh_lbs = weight_volume.forward_weight(cano_mesh_v.to(config.device)).cpu()[0]
    # posed_mesh_v_query = skinning(cano_mesh_v[None], cano_mesh_lbs[None], jnt_mats)
    # posed_mesh_query = trimesh.Trimesh(posed_mesh_v_query[0].cpu().numpy(), cano_mesh.faces, process = False)
    # posed_mesh_query.export('../debug/posed_mesh_query.obj')


if __name__ == '__main__':
    from argparse import ArgumentParser
    import yaml

    arg_parser = ArgumentParser()
    arg_parser.add_argument('-c', '--config_path', type = str, help = 'Configuration file path.')
    args = arg_parser.parse_args()

    opt = yaml.load(open(args.config_path, encoding = 'UTF-8'), Loader = yaml.FullLoader)
    data_dir = opt['train']['data']['data_dir']

    calc_cano_weight_volume(data_dir, gender = 'neutral')
