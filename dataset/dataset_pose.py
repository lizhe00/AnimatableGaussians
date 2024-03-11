import glob
import os
import pickle
import numpy as np
import cv2 as cv
import torch
import trimesh
from torch.utils.data import Dataset
import yaml
import json
import smplx

import dataset.commons as commons
import utils.nerf_util as nerf_util
import utils.visualize_util as visualize_util
import config


class PoseDataset(Dataset):
    @torch.no_grad()
    def __init__(
        self,
        data_path,
        frame_range = None,
        frame_interval = 1,
        smpl_shape = None,
        gender = 'neutral',
        frame_win = 0,
        fix_head_pose = True,
        fix_hand_pose = True,
        denoise = False,
        hand_pose_type = 'ori',
        constrain_leg_pose = False,
        device = 'cuda:0'
    ):
        super(PoseDataset, self).__init__()

        self.data_path = data_path
        self.training = False

        self.gender = gender

        data_name, ext = os.path.splitext(os.path.basename(data_path))
        if ext == '.pkl':
            smpl_data = pickle.load(open(data_path, 'rb'))
            smpl_data = dict(smpl_data)
            self.body_poses = torch.from_numpy(smpl_data['smpl_poses']).to(torch.float32)
            self.transl = torch.from_numpy(smpl_data['smpl_trans']).to(torch.float32) * 1e-3
            self.dataset_name = 'aist++'
            self.seq_name = data_name
        elif ext == '.npz':
            potential_datasets = ['thuman4', 'actorshq', 'avatarrex', 'AMASS']
            for i, potential_dataset in enumerate(potential_datasets):
                start_pos = data_path.find(potential_dataset)
                if start_pos == -1:
                    if i < len(potential_datasets) - 1:
                        continue
                    else:
                        raise ValueError('Invalid data_path!')
                self.dataset_name = potential_dataset
                self.seq_name = data_path[start_pos:].replace(self.dataset_name, '').replace('/', '_').replace('\\', '_').replace('.npz', '')
                break

            if self.dataset_name == 'thuman4' or self.dataset_name == 'actorshq' or self.dataset_name == 'avatarrex':
                smpl_data = np.load(data_path)
                smpl_data = dict(smpl_data)
            else:  # AMASS dataset
                pose_file = np.load(data_path)
                smpl_data = {
                    'betas': np.zeros((1, 10), np.float32),
                    'global_orient': pose_file['poses'][:, :3],
                    'transl': pose_file['trans'],
                    'body_pose': pose_file['poses'][:, 3: 22 * 3],
                    'left_hand_pose': pose_file['poses'][:, 22 * 3: 37 * 3],
                    'right_hand_pose': pose_file['poses'][:, 37 * 3:]
                }

                smpl_data['body_pose'][:, 13 * 3 + 2] -= 0.3
                smpl_data['body_pose'][:, 12 * 3 + 2] += 0.3
                # smpl_data['body_pose'][:, 16 * 3 + 2] -= 0.1
                # smpl_data['body_pose'][:, 15 * 3 + 2] += 0.1
                smpl_data['body_pose'][:, 19 * 3: 20 * 3] = 0.
                smpl_data['body_pose'][:, 20 * 3: 21 * 3] = 0.
                smpl_data['body_pose'][:, 14 * 3] = 0.

            if self.seq_name == '_actor01':
                smpl_data['body_pose'][:, 6*3: 7*3] = 0.
                smpl_data['body_pose'][:, 7*3: 8*3] = 0.

            smpl_data = {k: torch.from_numpy(v).to(torch.float32) for k, v in smpl_data.items()}
            frame_num = smpl_data['body_pose'].shape[0]
            self.body_poses = torch.zeros((frame_num, 72), dtype = torch.float32)
            self.body_poses[:, :3] = smpl_data['global_orient']
            self.body_poses[:, 3:3+21*3] = smpl_data['body_pose']
            self.transl = smpl_data['transl']

            data_dir = os.path.dirname(data_path)
            calib_path = os.path.basename(data_path).replace('.npz', '.json').replace('pose', 'calibration')
            calib_path = data_dir + '/' + calib_path
            if os.path.exists(calib_path):
                cam_data = json.load(open(calib_path, 'r'))
                self.view_num = len(cam_data)
                self.extr_mats = []
                self.cam_names = list(cam_data.keys())
                for view_idx in range(self.view_num):
                    extr_mat = np.identity(4, np.float32)
                    extr_mat[:3, :3] = np.array(cam_data[self.cam_names[view_idx]]['R'], np.float32).reshape(3, 3)
                    extr_mat[:3, 3] = np.array(cam_data[self.cam_names[view_idx]]['T'], np.float32)
                    self.extr_mats.append(extr_mat)
                self.intr_mats = [np.array(cam_data[self.cam_names[view_idx]]['K'], np.float32).reshape(3, 3) for view_idx in range(self.view_num)]
                self.img_heights = [cam_data[self.cam_names[view_idx]]['imgSize'][1] for view_idx in range(self.view_num)]
                self.img_widths = [cam_data[self.cam_names[view_idx]]['imgSize'][0] for view_idx in range(self.view_num)]
        else:
            raise AssertionError('Invalid data_path!')

        if 'left_hand_pose' in smpl_data:
            self.left_hand_pose = smpl_data['left_hand_pose']
        else:
            self.left_hand_pose = config.left_hand_pose[None].expand(self.body_poses.shape[0], -1)
        if 'right_hand_pose' in smpl_data:
            self.right_hand_pose = smpl_data['right_hand_pose']
        else:
            self.right_hand_pose = config.right_hand_pose[None].expand(self.body_poses.shape[0], -1)

        self.body_poses = self.body_poses.to(device)
        self.transl = self.transl.to(device)

        self.fix_head_pose = fix_head_pose
        self.fix_hand_pose = fix_hand_pose

        self.smpl_model = smplx.SMPLX(model_path = config.PROJ_DIR + '/smpl_files/smplx', gender = self.gender, use_pca = False, num_pca_comps = 45, flat_hand_mean = True, batch_size = 1).to(device)

        pose_list = list(range(0, self.body_poses.shape[0], frame_interval))
        if frame_range is not None:
            if isinstance(frame_range, list):
                if isinstance(frame_range[0], list):
                    self.pose_list = []
                    for interval in frame_range:
                        if len(interval) == 2 or len(interval) == 3:
                            self.pose_list += list(range(*interval))
                        else:
                            for i in range(interval[3]):
                                self.pose_list += list(range(interval[0], interval[1], interval[2]))
                else:
                    if len(frame_range) == 2:
                        print(f'# Selected frame indices: range({frame_range[0]}, {frame_range[1]})')
                        frame_range = range(frame_range[0], frame_range[1])
                    elif len(frame_range) == 3:
                        print(f'# Selected frame indices: range({frame_range[0]}, {frame_range[1]}, {frame_range[2]})')
                        frame_range = range(frame_range[0], frame_range[1], frame_range[2])
                    self.pose_list = list(frame_range)
        else:
            self.pose_list = pose_list

        print('# Pose list: ', self.pose_list)
        print('# Dataset contains %d items' % len(self))

        # SMPL related
        self.smpl_shape = smpl_shape.to(torch.float32).to(device) if smpl_shape is not None else torch.zeros(10, dtype = torch.float32)
        ret = self.smpl_model.forward(betas = self.smpl_shape[None],
                                      global_orient = config.cano_smpl_global_orient[None].to(device),
                                      transl = config.cano_smpl_transl[None].to(device),
                                      body_pose = config.cano_smpl_body_pose[None].to(device),
                                      # left_hand_pose = config.left_hand_pose[None],
                                      # right_hand_pose = config.right_hand_pose[None]
                                      )
        self.cano_smpl = {k: v[0] for k, v in ret.items() if isinstance(v, torch.Tensor)}
        self.inv_cano_jnt_mats = torch.linalg.inv(self.cano_smpl['A'])
        min_xyz = self.cano_smpl['vertices'].min(0)[0]
        max_xyz = self.cano_smpl['vertices'].max(0)[0]
        self.cano_smpl_center = 0.5 * (min_xyz + max_xyz)
        min_xyz[:2] -= 0.05
        max_xyz[:2] += 0.05
        min_xyz[2] -= 0.15
        max_xyz[2] += 0.15
        self.cano_bounds = torch.stack([min_xyz, max_xyz], 0).to(torch.float32).cpu().numpy()
        self.smpl_faces = self.smpl_model.faces.astype(np.int32)

        self.frame_win = int(frame_win)
        self.denoise = denoise
        if self.denoise:
            win_size = 1
            body_poses_clone = self.body_poses.clone()
            transl_clone = self.transl.clone()
            frame_num = body_poses_clone.shape[0]
            self.body_poses[win_size: frame_num-win_size] = 0
            self.transl[win_size: frame_num-win_size] = 0
            for i in range(-win_size, win_size + 1):
                self.body_poses[win_size: frame_num-win_size] += body_poses_clone[win_size+i: frame_num-win_size+i]
                self.transl[win_size: frame_num-win_size] += transl_clone[win_size+i: frame_num-win_size+i]
            self.body_poses[win_size: frame_num-win_size] /= (2 * win_size + 1)
            self.transl[win_size: frame_num-win_size] /= (2 * win_size + 1)

        self.hand_pose_type = hand_pose_type

        self.device = device
        self.last_data_idx = 0

        commons._initialize_hands(self)
        self.left_cano_mano_v, self.left_cano_mano_n, self.right_cano_mano_v, self.right_cano_mano_n \
            = commons.generate_two_manos(self, self.cano_smpl['vertices'])

        if constrain_leg_pose:
            # a = 14.
            # # print(self.body_poses[284, 1*3:2*3])
            # # print(self.body_poses[284, 2*3:3*3])
            # self.body_poses[:, 1*3] = torch.clip(self.body_poses[:, 1 * 3], -np.pi / a, np.pi / a)
            # self.body_poses[:, 2*3] = torch.clip(self.body_poses[:, 2 * 3], -np.pi / a, np.pi / a)
            # self.body_poses[:, 1 * 3+2] = torch.clip(self.body_poses[:, 1 * 3+2], -np.pi / a, np.pi / a)
            # self.body_poses[:, 2 * 3+2] = torch.clip(self.body_poses[:, 2 * 3+2], -np.pi / a, np.pi / a)
            # exit(1)

            self.body_poses[:, 4*3] = torch.clip(self.body_poses[:, 4*3], -0.3, 0.3)
            self.body_poses[:, 5*3] = torch.clip(self.body_poses[:, 5*3], -0.3, 0.3)

    def __len__(self):
        return len(self.pose_list)

    def __getitem__(self, index):
        return self.getitem(index)

    @torch.no_grad()
    def getitem(self, index, **kwargs):
        pose_idx = self.pose_list[index]
        if pose_idx == 0 or pose_idx > self.pose_list[min(index - 1, 0)]:
            data_idx = pose_idx
        else:
            data_idx = self.last_data_idx + 1
        # print('data index: %d, pose index: %d' % (data_idx, pose_idx))

        if self.hand_pose_type == 'fist':
            left_hand_pose = config.left_hand_pose.to(self.device).clone()
            right_hand_pose = config.right_hand_pose.to(self.device).clone()
            left_hand_pose[:3] = 0.
            right_hand_pose[:3] = 0.
        elif self.hand_pose_type == 'normal':
            left_hand_pose = torch.tensor([0.10859203338623047, 0.10181399434804916, -0.2822268009185791, 0.10211331397294998, -0.09689036756753922, -0.4484838545322418, -0.11360692232847214, -0.023141659796237946, 0.10571160167455673, -0.08793719857931137, -0.026760095730423927, -0.41390693187713623, -0.0923849567770958, 0.10266668349504471, -0.36039748787879944, 0.02140655182301998, -0.07156527787446976, -0.04903153330087662, -0.22358819842338562, -0.3716682195663452, -0.2683027982711792, -0.1506909281015396, 0.07079305499792099, -0.34404537081718445, -0.168443500995636, -0.014021224342286587, 0.09489774703979492, -0.050323735922575, -0.18992969393730164, -0.43895423412323, -0.1806418001651764, 0.0198075994849205, -0.25444355607032776, -0.10171788930892944, -0.10680688172578812, -0.09953738003969193, 0.8094075918197632, 0.5156061053276062, -0.07900168001651764, -0.45094889402389526, 0.24947893619537354, 0.23369410634040833, 0.45277315378189087, -0.17375235259532928, -0.3077943027019501], dtype = torch.float32, device = self.device)
            right_hand_pose = torch.tensor([0.06415501981973648, -0.06942438334226608, 0.282951682806015, 0.09073827415704727, 0.0775153785943985, 0.2961004376411438, -0.07659692317247391, 0.004730052314698696, -0.12084470689296722, 0.007974660955369473, 0.05222926288843155, 0.32775357365608215, -0.10166633129119873, -0.06862349808216095, 0.174485981464386, -0.0023323255591094494, 0.04998664930462837, -0.03490559384226799, 0.12949667870998383, 0.26883721351623535, 0.06881044059991837, -0.18259745836257935, -0.08183271437883377, 0.17669665813446045, -0.08099694550037384, 0.04115655645728111, -0.17928685247898102, 0.07734024524688721, 0.13419172167778015, 0.2600148022174835, -0.151871919631958, -0.01772170141339302, 0.1267814189195633, -0.08800505846738815, 0.09480107575654984, 0.0016392067773267627, 0.6149336695671082, -0.32634419202804565, 0.02278662845492363, -0.39148610830307007, -0.22757330536842346, -0.07884717732667923, 0.38199105858802795, 0.13064607977867126, 0.20154500007629395], dtype = torch.float32, device = self.device)
        elif self.hand_pose_type == 'zero':
            left_hand_pose = torch.zeros(45, dtype = torch.float32, device = self.device)
            right_hand_pose = torch.zeros(45, dtype = torch.float32, device = self.device)
        elif self.hand_pose_type == 'ori':
            left_hand_pose = self.left_hand_pose[pose_idx].to(self.device)
            right_hand_pose = self.right_hand_pose[pose_idx].to(self.device)
        else:
            raise ValueError('Invalid hand_pose_type!')

        # SMPL
        live_smpl = self.smpl_model.forward(betas = self.smpl_shape[None],
                                            global_orient = self.body_poses[pose_idx, :3][None],
                                            transl = self.transl[pose_idx][None],
                                            body_pose = self.body_poses[pose_idx, 3: 66][None],
                                            left_hand_pose = left_hand_pose[None],
                                            right_hand_pose = right_hand_pose[None]
                                            )

        # live_smpl_trimesh = trimesh.Trimesh(vertices = live_smpl.vertices[0].cpu().numpy(), faces = self.smpl_model.faces, process = False)
        # live_smpl_trimesh.export('./debug/smpl_amass.ply')
        # exit(1)

        live_smpl_woRoot = self.smpl_model.forward(betas = self.smpl_shape[None],
                                            # global_orient = self.body_poses[pose_idx, :3][None],
                                            # transl = self.transl[pose_idx][None],
                                            body_pose = self.body_poses[pose_idx, 3: 66][None],
                                            # left_hand_pose = config.left_hand_pose[None],
                                            # right_hand_pose = config.right_hand_pose[None]
                                            )

        # cano_smpl = self.smpl_model.forward(betas=self.smpl_shape[None],
        #                                     global_orient=config.cano_smpl_global_orient[None],
        #                                     transl=config.cano_smpl_transl[None],
        #                                     body_pose=config.cano_smpl_body_pose[None],
        #                                     # left_hand_pose = left_hand_pose[None],
        #                                     # right_hand_pose = right_hand_pose[None]
        #                                     )

        data_item = dict()
        data_item['item_idx'] = index
        data_item['data_idx'] = data_idx
        data_item['global_orient'] = self.body_poses[pose_idx, :3]
        data_item['transl'] = self.transl[pose_idx]
        data_item['joints'] = live_smpl.joints[0, :22]
        data_item['kin_parent'] = self.smpl_model.parents[:22].to(torch.long)
        data_item['pose_1st'] = self.body_poses[0, 3: 66]
        if self.frame_win > 0:
            total_frame_num = len(self.pose_list)
            selected_frames = self.pose_list[max(0, index - self.frame_win): min(total_frame_num, index + self.frame_win + 1)]
            data_item['pose'] = self.body_poses[selected_frames, 3: 66].clone()
        else:
            data_item['pose'] = self.body_poses[pose_idx, 3: 66].clone()

        if self.fix_head_pose:
            data_item['pose'][..., 3 * 11: 3 * 11 + 3] = 0.
            data_item['pose'][..., 3 * 14: 3 * 14 + 3] = 0.
        if self.fix_hand_pose:
            data_item['pose'][..., 3 * 19: 3 * 19 + 3] = 0.
            data_item['pose'][..., 3 * 20: 3 * 20 + 3] = 0.
        data_item['lhand_pose'] = torch.zeros_like(config.left_hand_pose)
        data_item['rhand_pose'] = torch.zeros_like(config.right_hand_pose)
        data_item['time_stamp'] = np.array(pose_idx, np.float32)
        data_item['live_smpl_v'] = live_smpl.vertices[0]
        data_item['live_smpl_v_woRoot'] = live_smpl_woRoot.vertices[0]
        data_item['cano_smpl_v'] = self.cano_smpl['vertices']
        data_item['cano_jnts'] = self.cano_smpl['joints']
        inv_cano_jnt_mats = torch.linalg.inv(self.cano_smpl['A'])
        data_item['cano2live_jnt_mats'] = torch.matmul(live_smpl.A[0], inv_cano_jnt_mats)
        data_item['cano2live_jnt_mats_woRoot'] = torch.matmul(live_smpl_woRoot.A[0], inv_cano_jnt_mats)
        data_item['cano_smpl_center'] = self.cano_smpl_center
        data_item['cano_bounds'] = self.cano_bounds
        data_item['smpl_faces'] = self.smpl_faces
        min_xyz = live_smpl.vertices[0].min(0)[0] - 0.15
        max_xyz = live_smpl.vertices[0].max(0)[0] + 0.15
        live_bounds = torch.stack([min_xyz, max_xyz], 0).to(torch.float32).cpu().numpy()
        data_item['live_bounds'] = live_bounds

        # # mano
        # data_item['left_cano_mano_v'], data_item['left_cano_mano_n'], data_item['right_cano_mano_v'], data_item['right_cano_mano_n']\
        #     = commons.generate_two_manos(self, self.cano_smpl['vertices'])
        # data_item['left_live_mano_v'], data_item['left_live_mano_n'], data_item['right_live_mano_v'], data_item['right_live_mano_n'] \
        #     = commons.generate_two_manos(self, live_smpl.vertices[0])

        """ synthesis config """
        img_h = 512 if 'img_h' not in kwargs else kwargs['img_h']
        img_w = 512 if 'img_w' not in kwargs else kwargs['img_w']
        intr = np.array([[550, 0, 256], [0, 550, 256], [0, 0, 1]], np.float32) if 'intr' not in kwargs else kwargs['intr']
        if 'extr' not in kwargs:
            extr = visualize_util.calc_front_mv(live_bounds.mean(0), tar_pos = np.array([0, 0, 2.5]))
        else:
            extr = kwargs['extr']

        """ training data config of view_idx """
        # view_idx = 0
        # img_h = self.img_heights[view_idx]
        # img_w = self.img_widths[view_idx]
        # intr = self.intr_mats[view_idx]
        # extr = self.extr_mats[view_idx]

        uv = self.gen_uv(img_w, img_h)
        uv = uv.reshape(-1, 2)
        ray_d, ray_o = nerf_util.get_rays(uv, extr, intr)
        near, far, mask_at_bound = nerf_util.get_near_far(live_bounds, ray_o, ray_d)
        uv = uv[mask_at_bound]
        ray_o = ray_o[mask_at_bound]
        ray_d = ray_d[mask_at_bound]

        data_item.update({
            'uv': uv,
            'ray_o': ray_o,
            'ray_d': ray_d,
            'near': near,
            'far': far,
            'dist': np.zeros_like(near),
            'img_h': img_h,
            'img_w': img_w,
            'extr': extr,
            'intr': intr
        })

        return data_item

    def getitem_fast(self, index, **kwargs):
        pose_idx = self.pose_list[index]
        if pose_idx == 0 or pose_idx > self.last_data_idx:
            data_idx = pose_idx
        else:
            data_idx = self.last_data_idx + 1
        # print('data index: %d, pose index: %d' % (data_idx, pose_idx))

        if self.hand_pose_type == 'fist':
            left_hand_pose = config.left_hand_pose.to(self.device)
            right_hand_pose = config.right_hand_pose.to(self.device)
        elif self.hand_pose_type == 'normal':
            left_hand_pose = torch.tensor(
                [0.10859203338623047, 0.10181399434804916, -0.2822268009185791, 0.10211331397294998, -0.09689036756753922, -0.4484838545322418, -0.11360692232847214, -0.023141659796237946, 0.10571160167455673, -0.08793719857931137, -0.026760095730423927, -0.41390693187713623, -0.0923849567770958, 0.10266668349504471, -0.36039748787879944, 0.02140655182301998, -0.07156527787446976, -0.04903153330087662, -0.22358819842338562, -0.3716682195663452, -0.2683027982711792, -0.1506909281015396,
                 0.07079305499792099, -0.34404537081718445, -0.168443500995636, -0.014021224342286587, 0.09489774703979492, -0.050323735922575, -0.18992969393730164, -0.43895423412323, -0.1806418001651764, 0.0198075994849205, -0.25444355607032776, -0.10171788930892944, -0.10680688172578812, -0.09953738003969193, 0.8094075918197632, 0.5156061053276062, -0.07900168001651764, -0.45094889402389526, 0.24947893619537354, 0.23369410634040833, 0.45277315378189087, -0.17375235259532928,
                 -0.3077943027019501], dtype = torch.float32, device = self.device)
            right_hand_pose = torch.tensor(
                [0.06415501981973648, -0.06942438334226608, 0.282951682806015, 0.09073827415704727, 0.0775153785943985, 0.2961004376411438, -0.07659692317247391, 0.004730052314698696, -0.12084470689296722, 0.007974660955369473, 0.05222926288843155, 0.32775357365608215, -0.10166633129119873, -0.06862349808216095, 0.174485981464386, -0.0023323255591094494, 0.04998664930462837, -0.03490559384226799, 0.12949667870998383, 0.26883721351623535, 0.06881044059991837, -0.18259745836257935,
                 -0.08183271437883377, 0.17669665813446045, -0.08099694550037384, 0.04115655645728111, -0.17928685247898102, 0.07734024524688721, 0.13419172167778015, 0.2600148022174835, -0.151871919631958, -0.01772170141339302, 0.1267814189195633, -0.08800505846738815, 0.09480107575654984, 0.0016392067773267627, 0.6149336695671082, -0.32634419202804565, 0.02278662845492363, -0.39148610830307007, -0.22757330536842346, -0.07884717732667923, 0.38199105858802795, 0.13064607977867126,
                 0.20154500007629395], dtype = torch.float32, device = self.device)
        elif self.hand_pose_type == 'zero':
            left_hand_pose = torch.zeros(45, dtype = torch.float32, device = self.device)
            right_hand_pose = torch.zeros(45, dtype = torch.float32, device = self.device)
        elif self.hand_pose_type == 'ori':
            left_hand_pose = self.left_hand_pose[pose_idx].to(self.device)
            right_hand_pose = self.right_hand_pose[pose_idx].to(self.device)
        else:
            raise ValueError('Invalid hand_pose_type!')

        # SMPL
        live_smpl = self.smpl_model.forward(betas = self.smpl_shape[None],
                                            global_orient = self.body_poses[pose_idx, :3][None],
                                            transl = self.transl[pose_idx][None],
                                            body_pose = self.body_poses[pose_idx, 3: 66][None],
                                            left_hand_pose = left_hand_pose[None],
                                            right_hand_pose = right_hand_pose[None]
                                            )

        live_smpl_woRoot = self.smpl_model.forward(betas = self.smpl_shape[None],
                                                   # global_orient = self.body_poses[pose_idx, :3][None],
                                                   # transl = self.transl[pose_idx][None],
                                                   body_pose = self.body_poses[pose_idx, 3: 66][None],
                                                   # left_hand_pose = config.left_hand_pose[None],
                                                   # right_hand_pose = config.right_hand_pose[None]
                                                   )

        # cano_smpl = self.smpl_model.forward(betas = self.smpl_shape[None],
        #                                     global_orient = config.cano_smpl_global_orient[None],
        #                                     transl = config.cano_smpl_transl[None],
        #                                     body_pose = config.cano_smpl_body_pose[None],
        #                                     # left_hand_pose = left_hand_pose[None],
        #                                     # right_hand_pose = right_hand_pose[None]
        #                                     )

        data_item = dict()
        data_item['item_idx'] = index
        data_item['data_idx'] = data_idx
        data_item['global_orient'] = self.body_poses[pose_idx, :3]
        data_item['joints'] = live_smpl.joints[0, :22]
        data_item['kin_parent'] = self.smpl_model.parents[:22].to(torch.long)
        data_item['live_smpl_v'] = live_smpl.vertices[0]
        data_item['live_smpl_v_woRoot'] = live_smpl_woRoot.vertices[0]
        data_item['cano_smpl_v'] = self.cano_smpl['vertices']
        data_item['cano_jnts'] = self.cano_smpl['joints']
        inv_cano_jnt_mats = torch.linalg.inv(self.cano_smpl['A'])
        data_item['cano2live_jnt_mats'] = torch.matmul(live_smpl.A[0], inv_cano_jnt_mats)
        data_item['cano2live_jnt_mats_woRoot'] = torch.matmul(live_smpl_woRoot.A[0], inv_cano_jnt_mats)
        data_item['cano_smpl_center'] = self.cano_smpl_center
        data_item['cano_bounds'] = self.cano_bounds
        data_item['smpl_faces'] = self.smpl_faces
        min_xyz = live_smpl.vertices[0].min(0)[0] - 0.15
        max_xyz = live_smpl.vertices[0].max(0)[0] + 0.15
        live_bounds = torch.stack([min_xyz, max_xyz], 0).to(torch.float32).cpu().numpy()
        data_item['live_bounds'] = live_bounds

        data_item['left_cano_mano_v'], data_item['left_cano_mano_n'], data_item['right_cano_mano_v'], data_item['right_cano_mano_n'] \
            = self.left_cano_mano_v, self.left_cano_mano_n, self.right_cano_mano_v, self.right_cano_mano_n

        """ synthesis config """
        img_h = 512 if 'img_h' not in kwargs else kwargs['img_h']
        img_w = 512 if 'img_w' not in kwargs else kwargs['img_w']
        intr = np.array([[550, 0, 256], [0, 550, 256], [0, 0, 1]], np.float32) if 'intr' not in kwargs else kwargs['intr']
        if 'extr' not in kwargs:
            extr = visualize_util.calc_front_mv(live_bounds.mean(0), tar_pos = np.array([0, 0, 2.5]))
        else:
            extr = kwargs['extr']

        data_item.update({
            'img_h': img_h,
            'img_w': img_w,
            'extr': extr,
            'intr': intr
        })

        self.last_data_idx = data_idx

        return data_item

    def getitem_a_pose(self, **kwargs):
        hand_pose_type = 'fist'
        if hand_pose_type == 'fist':
            left_hand_pose = config.left_hand_pose.to(self.device)
            right_hand_pose = config.right_hand_pose.to(self.device)
        elif hand_pose_type == 'normal':
            left_hand_pose = torch.tensor(
                [0.10859203338623047, 0.10181399434804916, -0.2822268009185791, 0.10211331397294998, -0.09689036756753922, -0.4484838545322418, -0.11360692232847214, -0.023141659796237946, 0.10571160167455673, -0.08793719857931137, -0.026760095730423927, -0.41390693187713623, -0.0923849567770958, 0.10266668349504471, -0.36039748787879944, 0.02140655182301998, -0.07156527787446976, -0.04903153330087662, -0.22358819842338562, -0.3716682195663452, -0.2683027982711792, -0.1506909281015396,
                 0.07079305499792099, -0.34404537081718445, -0.168443500995636, -0.014021224342286587, 0.09489774703979492, -0.050323735922575, -0.18992969393730164, -0.43895423412323, -0.1806418001651764, 0.0198075994849205, -0.25444355607032776, -0.10171788930892944, -0.10680688172578812, -0.09953738003969193, 0.8094075918197632, 0.5156061053276062, -0.07900168001651764, -0.45094889402389526, 0.24947893619537354, 0.23369410634040833, 0.45277315378189087, -0.17375235259532928,
                 -0.3077943027019501], dtype = torch.float32, device = self.device)
            right_hand_pose = torch.tensor(
                [0.06415501981973648, -0.06942438334226608, 0.282951682806015, 0.09073827415704727, 0.0775153785943985, 0.2961004376411438, -0.07659692317247391, 0.004730052314698696, -0.12084470689296722, 0.007974660955369473, 0.05222926288843155, 0.32775357365608215, -0.10166633129119873, -0.06862349808216095, 0.174485981464386, -0.0023323255591094494, 0.04998664930462837, -0.03490559384226799, 0.12949667870998383, 0.26883721351623535, 0.06881044059991837, -0.18259745836257935,
                 -0.08183271437883377, 0.17669665813446045, -0.08099694550037384, 0.04115655645728111, -0.17928685247898102, 0.07734024524688721, 0.13419172167778015, 0.2600148022174835, -0.151871919631958, -0.01772170141339302, 0.1267814189195633, -0.08800505846738815, 0.09480107575654984, 0.0016392067773267627, 0.6149336695671082, -0.32634419202804565, 0.02278662845492363, -0.39148610830307007, -0.22757330536842346, -0.07884717732667923, 0.38199105858802795, 0.13064607977867126,
                 0.20154500007629395], dtype = torch.float32, device = self.device)
        elif self.hand_pose_type == 'zero':
            left_hand_pose = torch.zeros(45, dtype = torch.float32, device = self.device)
            right_hand_pose = torch.zeros(45, dtype = torch.float32, device = self.device)
        else:
            raise ValueError('Invalid hand_pose_type!')

        body_pose = torch.zeros(21 * 3, dtype = torch.float32).to(self.device)
        body_pose[15 * 3 + 2] += -0.8
        body_pose[16 * 3 + 2] += 0.8

        # SMPL
        live_smpl = self.smpl_model.forward(betas = self.smpl_shape[None],
                                            global_orient = None,
                                            transl = None,
                                            body_pose = body_pose[None],
                                            left_hand_pose = left_hand_pose[None],
                                            right_hand_pose = right_hand_pose[None]
                                            )

        live_smpl_woRoot = self.smpl_model.forward(betas = self.smpl_shape[None],
                                                   # global_orient = self.body_poses[pose_idx, :3][None],
                                                   # transl = self.transl[pose_idx][None],
                                                   body_pose = body_pose[None],
                                                   # left_hand_pose = config.left_hand_pose[None],
                                                   # right_hand_pose = config.right_hand_pose[None]
                                                   )

        # cano_smpl = self.smpl_model.forward(betas = self.smpl_shape[None],
        #                                     global_orient = config.cano_smpl_global_orient[None],
        #                                     transl = config.cano_smpl_transl[None],
        #                                     body_pose = config.cano_smpl_body_pose[None],
        #                                     # left_hand_pose = left_hand_pose[None],
        #                                     # right_hand_pose = right_hand_pose[None]
        #                                     )

        data_item = dict()
        data_item['item_idx'] = 0
        data_item['data_idx'] = 0
        data_item['global_orient'] = torch.zeros(3, dtype = torch.float32)
        data_item['joints'] = live_smpl.joints[0, :22]
        data_item['kin_parent'] = self.smpl_model.parents[:22].to(torch.long)
        data_item['live_smpl_v'] = live_smpl.vertices[0]
        data_item['live_smpl_v_woRoot'] = live_smpl_woRoot.vertices[0]
        data_item['cano_smpl_v'] = self.cano_smpl['vertices']
        data_item['cano_jnts'] = self.cano_smpl['joints']
        inv_cano_jnt_mats = torch.linalg.inv(self.cano_smpl['A'])
        data_item['cano2live_jnt_mats'] = torch.matmul(live_smpl.A[0], inv_cano_jnt_mats)
        data_item['cano2live_jnt_mats_woRoot'] = torch.matmul(live_smpl_woRoot.A[0], inv_cano_jnt_mats)
        data_item['cano_smpl_center'] = self.cano_smpl_center
        data_item['cano_bounds'] = self.cano_bounds
        data_item['smpl_faces'] = self.smpl_faces
        min_xyz = live_smpl.vertices[0].min(0)[0] - 0.15
        max_xyz = live_smpl.vertices[0].max(0)[0] + 0.15
        live_bounds = torch.stack([min_xyz, max_xyz], 0).to(torch.float32).cpu().numpy()
        data_item['live_bounds'] = live_bounds

        data_item['left_cano_mano_v'], data_item['left_cano_mano_n'], data_item['right_cano_mano_v'], data_item['right_cano_mano_n'] \
            = self.left_cano_mano_v, self.left_cano_mano_n, self.right_cano_mano_v, self.right_cano_mano_n

        """ synthesis config """
        img_h = 512 if 'img_h' not in kwargs else kwargs['img_h']
        img_w = 300 if 'img_w' not in kwargs else kwargs['img_w']
        intr = np.array([[550, 0, 150], [0, 550, 256], [0, 0, 1]], np.float32) if 'intr' not in kwargs else kwargs['intr']
        if 'extr' not in kwargs:
            extr = visualize_util.calc_front_mv(live_bounds.mean(0), tar_pos = np.array([0, 0, 2.5]))
        else:
            extr = kwargs['extr']

        data_item.update({
            'img_h': img_h,
            'img_w': img_w,
            'extr': extr,
            'intr': intr
        })

        return data_item

    @staticmethod
    def gen_uv(img_w, img_h):
        x, y = np.meshgrid(np.linspace(0, img_w - 1, img_w, dtype = np.int),
                           np.linspace(0, img_h - 1, img_h, dtype = np.int))
        uv = np.stack([x, y], axis = -1)
        return uv
