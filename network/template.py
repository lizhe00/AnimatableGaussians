import torch
import torch.nn as nn
import numpy as np
import pytorch3d.ops
import pytorch3d.transforms
import trimesh

import config
from network.mlp import MLPLinear, SdfMLP
from network.density import LaplaceDensity
from network.volume import CanoBlendWeightVolume
from network.hand_avatar import HandAvatar
from utils.embedder import get_embedder
import utils.nerf_util as nerf_util
import utils.smpl_util as smpl_util
import utils.geo_util as geo_util
from utils.posevocab_custom_ops.near_far_smpl import near_far_smpl
from utils.posevocab_custom_ops.nearest_face import nearest_face_pytorch3d
from utils.knn import knn_gather
import root_finding


class TemplateNet(nn.Module):
    def __init__(self, opt):
        super(TemplateNet, self).__init__()
        self.opt = opt

        self.pos_embedder, self.pos_dim = get_embedder(opt['multires'], 3)

        # canonical blend weight volume
        self.cano_weight_volume = CanoBlendWeightVolume(config.opt['train']['data']['data_dir'] + '/cano_weight_volume.npz')

        self.pose_feat_dim = 0

        """ geometry networks """
        geo_mlp_opt = {
            'in_channels': self.pos_dim + self.pose_feat_dim,
            'out_channels': 256 + 1,
            'inter_channels': [512, 256, 256, 256, 256, 256],
            'nlactv': nn.Softplus(beta = 100),
            'res_layers': [4],
            'geometric_init': True,
            'bias': 0.7,
            'weight_norm': True
        }
        self.geo_mlp = SdfMLP(**geo_mlp_opt)

        """ texture networks """
        if self.opt['use_viewdir']:
            self.viewdir_embedder, self.viewdir_dim = get_embedder(self.opt['multires_viewdir'], 3)
        else:
            self.viewdir_embedder, self.viewdir_dim = None, 0
        tex_mlp_opt = {
            'in_channels': 256 + self.viewdir_dim,
            'out_channels': 3,
            'inter_channels': [256, 256, 256],
            'nlactv': nn.ReLU(),
            'last_op': nn.Sigmoid()
        }
        self.tex_mlp = MLPLinear(**tex_mlp_opt)

        print('# MLPs: ')
        print(self.geo_mlp)
        print(self.tex_mlp)

        # sdf2density
        self.density_func = LaplaceDensity(params_init = {'beta': 0.01})

        # hand avatars
        self.with_hand = self.opt.get('with_hand', False)
        self.left_hand = HandAvatar()
        self.right_hand = HandAvatar()

        # for root finding
        from network.volume import compute_gradient_volume
        if self.opt.get('volume_type', 'diff') == 'diff':
            self.weight_volume = self.cano_weight_volume.diff_weight_volume[0].permute(1, 2, 3, 0).contiguous()
        else:
            self.weight_volume = self.cano_weight_volume.ori_weight_volume[0].permute(1, 2, 3, 0).contiguous()
        self.grad_volume = compute_gradient_volume(self.weight_volume.permute(3, 0, 1, 2), self.cano_weight_volume.voxel_size).permute(2, 3, 4, 0, 1)\
            .reshape(self.cano_weight_volume.res_x, self.cano_weight_volume.res_y, self.cano_weight_volume.res_z, -1).contiguous()
        self.res = torch.tensor([self.cano_weight_volume.res_x, self.cano_weight_volume.res_y, self.cano_weight_volume.res_z], dtype = torch.int32, device = config.device)

        self._initialize_hands()

    def _initialize_hands(self):
        smplx_lhand_to_mano_rhand_data = np.load(config.PROJ_DIR + '/smpl_files/mano/smplx_lhand_to_mano_rhand.npz', allow_pickle = True)
        smplx_rhand_to_mano_rhand_data = np.load(config.PROJ_DIR + '/smpl_files/mano/smplx_rhand_to_mano_rhand.npz', allow_pickle = True)
        smpl_lhand_vert_id = np.copy(smplx_lhand_to_mano_rhand_data['smpl_vert_id_to_mano'])
        smpl_rhand_vert_id = np.copy(smplx_rhand_to_mano_rhand_data['smpl_vert_id_to_mano'])
        self.smpl_lhand_vert_id = torch.from_numpy(smpl_lhand_vert_id).to(config.device)
        self.smpl_rhand_vert_id = torch.from_numpy(smpl_rhand_vert_id).to(config.device)
        self.smpl_hands_vert_id = torch.cat([self.smpl_lhand_vert_id, self.smpl_rhand_vert_id], 0)
        mano_face_closed = np.loadtxt(config.PROJ_DIR + '/smpl_files/mano/mano_face_close.txt').astype(np.int64)
        self.mano_face_closed = torch.from_numpy(mano_face_closed).to(config.device)
        self.mano_face_closed_2hand = torch.cat([self.mano_face_closed[:, [2, 1, 0]], self.mano_face_closed + self.smpl_lhand_vert_id.shape[0]], 0)

    def forward_cano_body_nerf(self, xyz, viewdirs, pose, compute_grad = False):
        """
        :param xyz: (B, N, 3)
        :param viewdirs: (B, N, 3)
        :param pose: (B, pose_dim)
        :param compute_grad: whether computing gradient w.r.t xyz
        :return:
        """
        if compute_grad:
            xyz.requires_grad_()
        # pose_feat = self.pose_feat[None, None].expand(xyz.shape[0], xyz.shape[1], -1)
        # pose_feat = torch.cat([self.pos_embedder(xyz), pose_feat], -1)
        pose_feat = self.pos_embedder(xyz)
        geo_feat = self.geo_mlp(pose_feat)
        sdf, geo_feat = torch.split(geo_feat, [1, geo_feat.shape[-1] - 1], -1)

        if self.viewdir_embedder is not None:
            if viewdirs is None:
                viewdirs = torch.zeros_like(xyz)
            geo_feat = torch.cat([geo_feat, self.viewdir_embedder(viewdirs)], -1)
        color = self.tex_mlp(geo_feat)

        density = self.density_func(sdf)

        ret = {
            'sdf': -sdf,  # assume outside is negative, inside is positive
            'density': density,
            'color': color,
            'cano_xyz': xyz.detach()
        }

        if compute_grad:
            d_output = torch.ones_like(sdf, requires_grad = False, device = sdf.device)
            normal = torch.autograd.grad(outputs = sdf,
                                         inputs = xyz,
                                         grad_outputs = d_output,
                                         create_graph = self.training,
                                         retain_graph = self.training,
                                         only_inputs = True)[0]
            ret.update({
                'normal': normal
            })
        return ret

    def forward_cano_hand_nerf(self, xyz, sdf, viewdirs, hand_pose, module = 'left_hand'):
        net = self.__getattr__(module)
        return net(xyz, sdf, viewdirs, hand_pose)

    def fuse_hands(self, body_ret, posed_xyz, view_dirs, batch, space = 'live'):
        # get hand correspondences
        batch_size, n_pts = posed_xyz.shape[:2]

        def process_one_hand(side = 'left'):
            hand_v = batch['%s_live_mano_v' % side] if space == 'live' else batch['%s_cano_mano_v' % side]
            hand_n = batch['%s_live_mano_n' % side] if space == 'live' else batch['%s_cano_mano_n' % side]
            hand_f = self.mano_face_closed[:, [2, 1, 0]] if side == 'left' else self.mano_face_closed

            dists, face_indices, bc_coords = nearest_face_pytorch3d(posed_xyz, hand_v, hand_f)
            face_vertex_ids = torch.gather(hand_f[None].expand(batch_size, -1, -1), 1, face_indices[:, :, None].long().expand(-1, -1, 3))  # (B, N, 3)

            cano_hand_v = geo_util.normalize_vert_bbox(batch['%s_cano_mano_v' % side], dim = 1, per_axis = True)

            face_cano_mano_v = knn_gather(cano_hand_v, face_vertex_ids)
            pts_cano_mano_v = (bc_coords[..., None] * face_cano_mano_v).sum(2)

            face_live_mano_v = knn_gather(hand_v, face_vertex_ids)
            pts_live_mano_v = (bc_coords[..., None] * face_live_mano_v).sum(2)

            # face_normal = torch.cross(face_live_smpl_v[:, :, 1] - face_live_smpl_v[:, :, 0], face_live_smpl_v[:, :, 2] - face_live_smpl_v[:, :, 0])
            face_live_mano_n = knn_gather(hand_n, face_vertex_ids)
            pts_live_mano_n = (bc_coords[..., None] * face_live_mano_n).sum(2)

            pts_smpl_sdf = -torch.sign(torch.einsum('bni,bni->bn', pts_live_mano_n, posed_xyz - pts_live_mano_v)) * dists

            return pts_cano_mano_v, pts_smpl_sdf.unsqueeze(-1)

        left_cano_mano_v, left_mano_sdf = process_one_hand('left')
        right_cano_mano_v, right_mano_sdf = process_one_hand('right')

        # fuse
        zero_hand_pose = torch.zeros((1, 15*3)).to(left_cano_mano_v)
        color_lhand = self.forward_cano_hand_nerf(left_cano_mano_v, left_mano_sdf, view_dirs, zero_hand_pose, module = 'left_hand')
        color_rhand = self.forward_cano_hand_nerf(right_cano_mano_v, right_mano_sdf, view_dirs, zero_hand_pose, module = 'right_hand')

        # calculate the blending weights for blending the outputs of body network and hand networks
        # wl = torch.sigmoid(1000 * (left_mano_sdf + 0.1)) * torch.sigmoid(25 * (left_cano_mano_v[..., 0:1] + 0.8))
        # wr = torch.sigmoid(1000 * (right_mano_sdf + 0.1)) * torch.sigmoid(-25 * (right_cano_mano_v[..., 0:1] - 0.8))
        cano_xyz = body_ret['cano_xyz']
        wl = torch.sigmoid(25 * (geo_util.normalize_vert_bbox(batch['left_cano_mano_v'], attris = cano_xyz, dim = 1, per_axis = True)[..., 0:1] + 0.8))
        wr = torch.sigmoid(-25 * (geo_util.normalize_vert_bbox(batch['right_cano_mano_v'], attris = cano_xyz, dim = 1, per_axis = True)[..., 0:1] - 0.8))
        wl[cano_xyz[..., 1] < batch['cano_smpl_center'][0, 1]] = 0.
        wr[cano_xyz[..., 1] < batch['cano_smpl_center'][0, 1]] = 0.

        s = torch.maximum(wl + wr, torch.ones_like(wl))
        wl, wr = wl / s, wr / s

        # blend the outputs of body network and hand networks
        w = wl + wr
        # factor = 10
        # left_mano_sdf *= factor
        # right_mano_sdf *= factor
        body_ret['sdf'] = wl * left_mano_sdf + wr * right_mano_sdf + (1.0 - w) * body_ret['sdf']
        body_ret['color'] = wl * color_lhand + wr * color_rhand + (1.0 - w) * body_ret['color']

        body_ret['density'] = self.density_func(-body_ret['sdf'])

    def forward_cano_radiance_field(self, xyz, view_dirs, batch):
        body_ret = self.forward_cano_body_nerf(xyz, view_dirs, None, compute_grad = self.training)

        return body_ret

    def transform_cano2live(self, cano_pts, batch, normals = None, near_thres = 0.08):
        cano2live_jnt_mats = batch['cano2live_jnt_mats'].clone()
        if not self.with_hand:
            # make sure the hand transformation is totally rigid
            cano2live_jnt_mats[:, 25: 40] = cano2live_jnt_mats[:, 20: 21]
            cano2live_jnt_mats[:, 40: 55] = cano2live_jnt_mats[:, 21: 22]

        pts_w = self.cano_weight_volume.forward_weight(cano_pts)
        pt_mats = torch.einsum('bnj,bjxy->bnxy', pts_w, cano2live_jnt_mats)
        posed_pts = torch.einsum('bnxy,bny->bnx', pt_mats[..., :3, :3], cano_pts) + pt_mats[..., :3, 3]

        if normals is None:
            return posed_pts
        else:
            posed_normals = torch.einsum('bnxy,bny->bnx', pt_mats[..., :3, :3], normals)
            return posed_pts, posed_normals

    def transform_live2cano(self, posed_pts, batch, normals = None, near_thres = 0.08):
        cano2live_jnt_mats = batch['cano2live_jnt_mats'].clone()
        if not self.with_hand:
            cano2live_jnt_mats[:, 25: 40] = cano2live_jnt_mats[:, 20: 21]
            cano2live_jnt_mats[:, 40: 55] = cano2live_jnt_mats[:, 21: 22]

        """ live_pts -> cano_pts """
        batch_size, n_pts = posed_pts.shape[:2]
        with torch.no_grad():
            if 'live_mesh_v' in batch:
            # if False:
                tar_v = batch['live_mesh_v']
                tar_f = batch['live_mesh_f']
                tar_lbs = batch['live_mesh_lbs']
                pts_w, near_flag = smpl_util.calc_blending_weight(posed_pts, tar_v, tar_f, tar_lbs, near_thres, method = 'NN')
            else:
                tar_v = batch['live_smpl_v']
                tar_f = batch['smpl_faces']
                tar_lbs = None
                pts_w, near_flag = smpl_util.calc_blending_weight(posed_pts, tar_v, tar_f, tar_lbs, near_thres, method = 'barycentric')

            pt_mats = torch.einsum('bnj,bjxy->bnxy', pts_w, cano2live_jnt_mats)
            pt_mats = torch.linalg.inv(pt_mats)
            cano_pts = torch.einsum('bnxy,bny->bnx', pt_mats[..., :3, :3], posed_pts) + pt_mats[..., :3, 3]
            # cano_pts_bk = cano_pts.detach().clone()

            if normals is not None:
                cano_normals = torch.einsum('bnxy,bny->bnx', pt_mats[..., :3, :3], normals)

        if self.opt['use_root_finding']:
            argmax_lbs = torch.argmax(pts_w, -1)
            nonopt_bone_ids = [7, 8, 10, 11]
            nonopt_pts_flag = torch.zeros((batch_size, n_pts), dtype = torch.bool).to(argmax_lbs.device)
            for i in nonopt_bone_ids:
                nonopt_pts_flag = torch.logical_or(nonopt_pts_flag, argmax_lbs == i)
            root_finding_flag = torch.logical_not(nonopt_pts_flag)
            if root_finding_flag.any():
                cano_pts_ = cano_pts[root_finding_flag].unsqueeze(0)
                posed_pts_ = posed_pts[root_finding_flag].unsqueeze(0)
                if not cano_pts_.is_contiguous():
                    cano_pts_ = cano_pts_.contiguous()
                if not posed_pts_.is_contiguous():
                    posed_pts_ = posed_pts_.contiguous()
                root_finding.root_finding(
                    self.weight_volume,
                    self.grad_volume,
                    posed_pts_,
                    cano_pts_,
                    cano2live_jnt_mats,
                    self.cano_weight_volume.volume_bounds,
                    self.res,
                    cano_pts_,
                    0.1,
                    10
                )
                cano_pts[root_finding_flag] = cano_pts_[0]

        if normals is None:
            return cano_pts, near_flag
        else:
            return cano_pts, cano_normals, near_flag

    def render(self, batch, chunk_size = 2048, depth_guided_sampling = None, space = 'live', white_bkgd = False):
        ray_o = batch['ray_o']
        ray_d = batch['ray_d']
        near = batch['near']
        far = batch['far']

        if depth_guided_sampling['flag']:
            print('# depth-guided sampling')
            valid_dist_flag = batch['dist'] > 1e-6
            dist = batch['dist'][valid_dist_flag]
            near_dist = depth_guided_sampling['near_sur_dist']
            far_dist = depth_guided_sampling['near_sur_dist']
            near[valid_dist_flag] = dist - near_dist
            far[valid_dist_flag] = dist + far_dist
            N_ray_samples = depth_guided_sampling['N_ray_samples']
        else:
            if depth_guided_sampling.get('type', 'smpl') == 'smpl':
                print('# smpl-guided sampling')
                valid_dist_flag = torch.ones_like(near, dtype = bool)
                near, far, intersect_flag = near_far_smpl(batch['live_smpl_v'][0], ray_o[0], ray_d[0])
                near[~intersect_flag] = batch['near'][0][~intersect_flag]
                far[~intersect_flag] = batch['far'][0][~intersect_flag]
                near = near.unsqueeze(0)
                far = far.unsqueeze(0)
                N_ray_samples = 64
            elif depth_guided_sampling.get('type', 'smpl') == 'uniform':
                print('# uniform sampling')
                valid_dist_flag = torch.ones_like(near, dtype = bool)
                N_ray_samples = 64

        if self.training:
            chunk_size = batch['ray_o'].shape[1]

        batch_size, n_pixels = ray_o.shape[:2]
        output_list = []
        for i in range(0, n_pixels, chunk_size):
            near_chunk = near[:, i: i + chunk_size]
            far_chunk = far[:, i: i + chunk_size]
            ray_o_chunk = ray_o[:, i: i + chunk_size]
            ray_d_chunk = ray_d[:, i: i + chunk_size]
            valid_dist_flag_chunk = valid_dist_flag[:, i: i + chunk_size]

            # sample points on each ray
            pts, z_vals = nerf_util.sample_pts_on_rays(ray_o_chunk, ray_d_chunk, near_chunk, far_chunk,
                                                       N_samples = N_ray_samples,
                                                       perturb = self.training,
                                                       depth_guided_mask = valid_dist_flag_chunk)

            # # debug: visualize pts
            # import trimesh
            # pts_trimesh = trimesh.PointCloud(pts[0].cpu().numpy().reshape(-1, 3))
            # pts_trimesh.export('./debug/sampled_pts_%s.obj' % 'training' if self.training else 'testing')
            # exit(1)

            # flat
            _, n_pixels_chunk, n_samples = pts.shape[:3]
            pts = pts.view(batch_size, n_pixels_chunk * n_samples, -1)
            dists = z_vals[..., 1:] - z_vals[..., :-1]
            dists = torch.cat([dists, dists[..., -1:]], -1)

            # query
            if space == 'live':
                cano_pts, near_flag = self.transform_live2cano(pts, batch)
            elif space == 'cano':
                cano_pts = pts
            else:
                raise ValueError('Invalid rendering space!')
            viewdirs = ray_d_chunk / torch.norm(ray_d_chunk, dim = -1, keepdim = True)
            viewdirs = viewdirs[:, :, None, :].expand(-1, -1, n_samples, -1).reshape(batch_size, n_pixels_chunk * n_samples, -1)
            # apply gaussian noise to avoid overfitting
            if self.training:
                with torch.no_grad():
                    noise = torch.randn_like(viewdirs) * 0.1
                viewdirs = viewdirs + noise
                viewdirs = viewdirs / torch.norm(viewdirs, dim = -1, keepdim = True)

            ret = self.forward_cano_radiance_field(cano_pts, viewdirs, batch)
            if self.with_hand:
                self.fuse_hands(ret, pts, viewdirs, batch, space)

            ret['color'] = ret['color'].view(batch_size, n_pixels_chunk, n_samples, -1)
            ret['density'] = ret['density'].view(batch_size, n_pixels_chunk, n_samples, -1)

            # integration
            alpha = 1. - torch.exp(-ret['density'] * dists[..., None])
            raw = torch.cat([ret['color'], alpha], dim = -1)
            rgb_map, disp_map, acc_map, weights, depth_map = nerf_util.raw2outputs(raw, z_vals, white_bkgd = white_bkgd)

            output_chunk = {
                'rgb_map': rgb_map,  # (batch_size, n_pixel_chunk, 3)
                'acc_map': acc_map
            }
            if 'normal' in ret:
                output_chunk.update({
                    'normal': ret['normal'].view(batch_size, n_pixels_chunk, -1, 3)
                })
            if 'tv_loss' in ret:
                output_chunk.update({
                    'tv_loss': ret['tv_loss'].view(1, 1, -1)
                })
            output_list.append(output_chunk)

        keys = output_list[0].keys()
        output_list = {k: torch.cat([r[k] for r in output_list], dim = 1) for k in keys}

        # processing for patch-based ray sampling
        if 'mask_within_patch' in batch:
            _, ray_num = batch['mask_within_patch'].shape
            rgb_map = torch.zeros((batch_size, ray_num, 3), dtype = torch.float32, device = config.device)
            acc_map = torch.zeros((batch_size, ray_num), dtype = torch.float32, device = config.device)
            rgb_map[batch['mask_within_patch']] = output_list['rgb_map'].reshape(-1, 3)
            acc_map[batch['mask_within_patch']] = output_list['acc_map'].reshape(-1)
            batch['color_gt'][~batch['mask_within_patch']] = 0.
            batch['mask_gt'][~batch['mask_within_patch']] = 0.
            output_list['rgb_map'] = rgb_map
            output_list['acc_map'] = acc_map

        return output_list
