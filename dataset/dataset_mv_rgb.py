import glob
import os
import numpy as np
import cv2 as cv
import torch
from torch.utils.data import Dataset

import smplx
import config
import utils.nerf_util as nerf_util
import utils.visualize_util as visualize_util
import dataset.commons as commons


class MvRgbDatasetBase(Dataset):
    @torch.no_grad()
    def __init__(
        self,
        data_dir,
        frame_range = None,
        used_cam_ids = None,
        training = True,
        subject_name = None,
        load_smpl_pos_map = False,
        load_smpl_nml_map = False,
        mode = '3dgs'
    ):
        super(MvRgbDatasetBase, self).__init__()

        self.data_dir = data_dir
        self.training = training
        self.subject_name = subject_name
        if self.subject_name is None:
            self.subject_name = os.path.basename(self.data_dir)
        self.load_smpl_pos_map = load_smpl_pos_map
        self.load_smpl_nml_map = load_smpl_nml_map
        self.mode = mode  # '3dgs' or 'nerf'

        self.load_cam_data()
        self.load_smpl_data()

        self.smpl_model = smplx.SMPLX(model_path = config.PROJ_DIR + '/smpl_files/smplx', gender = 'neutral', use_pca = False, num_pca_comps = 45, flat_hand_mean = True, batch_size = 1)

        pose_list = list(range(self.smpl_data['body_pose'].shape[0]))
        if frame_range is not None:
            if isinstance(frame_range, list):
                if len(frame_range) == 2:
                    print(f'# Selected frame indices: range({frame_range[0]}, {frame_range[1]})')
                    frame_range = range(frame_range[0], frame_range[1])
                elif len(frame_range) == 3:
                    print(f'# Selected frame indices: range({frame_range[0]}, {frame_range[1]}, {frame_range[2]})')
                    frame_range = range(frame_range[0], frame_range[1], frame_range[2])
            elif isinstance(frame_range, str):
                frame_range = np.loadtxt(self.data_dir + '/' + frame_range).astype(np.int).tolist()
                print(f'# Selected frame indices: {frame_range}')
            else:
                raise TypeError('Invalid frame_range!')
            self.pose_list = list(frame_range)
        else:
            self.pose_list = pose_list

        if self.training:
            if used_cam_ids is None:
                self.used_cam_ids = list(range(self.view_num))
            else:
                self.used_cam_ids = used_cam_ids
            print('# Used camera ids: ', self.used_cam_ids)
            self.data_list = []
            for pose_idx in self.pose_list:
                for view_idx in self.used_cam_ids:
                    self.data_list.append((pose_idx, view_idx))
            # filter missing files
            self.filter_missing_files()

        print('# Dataset contains %d items' % len(self))

        # SMPL related
        ret = self.smpl_model.forward(betas = self.smpl_data['betas'][0][None],
                                      global_orient = config.cano_smpl_global_orient[None],
                                      transl = config.cano_smpl_transl[None],
                                      body_pose = config.cano_smpl_body_pose[None])
        self.cano_smpl = {k: v[0] for k, v in ret.items() if isinstance(v, torch.Tensor)}
        self.inv_cano_jnt_mats = torch.linalg.inv(self.cano_smpl['A'])
        min_xyz = self.cano_smpl['vertices'].min(0)[0]
        max_xyz = self.cano_smpl['vertices'].max(0)[0]
        self.cano_smpl_center = 0.5 * (min_xyz + max_xyz)
        min_xyz[:2] -= 0.05
        max_xyz[:2] += 0.05
        min_xyz[2] -= 0.15
        max_xyz[2] += 0.15
        self.cano_bounds = torch.stack([min_xyz, max_xyz], 0).to(torch.float32).numpy()
        self.smpl_faces = self.smpl_model.faces.astype(np.int32)

        commons._initialize_hands(self)

    def __len__(self):
        if self.training:
            return len(self.data_list)
        else:
            return len(self.pose_list)

    def __getitem__(self, index):
        return self.getitem(index, self.training)

    def getitem(self, index, training = True, **kwargs):
        if training or kwargs.get('eval', False):  # training or evaluation
            pose_idx, view_idx = self.data_list[index]
            pose_idx = kwargs['pose_idx'] if 'pose_idx' in kwargs else pose_idx
            view_idx = kwargs['view_idx'] if 'view_idx' in kwargs else view_idx
            data_idx = (pose_idx, view_idx)
            if not training:
                print('data index: (%d, %d)' % (pose_idx, view_idx))
        else:  # testing
            pose_idx = self.pose_list[index]
            data_idx = pose_idx
            print('data index: %d' % pose_idx)

        # SMPL
        with torch.no_grad():
            live_smpl = self.smpl_model.forward(
                betas = self.smpl_data['betas'][0][None],
                global_orient = self.smpl_data['global_orient'][pose_idx][None],
                transl = self.smpl_data['transl'][pose_idx][None],
                body_pose = self.smpl_data['body_pose'][pose_idx][None],
                jaw_pose = self.smpl_data['jaw_pose'][pose_idx][None],
                expression = self.smpl_data['expression'][pose_idx][None],
                left_hand_pose = self.smpl_data['left_hand_pose'][pose_idx][None],
                right_hand_pose = self.smpl_data['right_hand_pose'][pose_idx][None]
            )
            cano_smpl = self.smpl_model.forward(
                betas = self.smpl_data['betas'][0][None],
                global_orient = config.cano_smpl_global_orient[None],
                transl = config.cano_smpl_transl[None],
                body_pose = config.cano_smpl_body_pose[None],
                jaw_pose = self.smpl_data['jaw_pose'][pose_idx][None],
                expression = self.smpl_data['expression'][pose_idx][None],
            )
            live_smpl_woRoot = self.smpl_model.forward(
                betas = self.smpl_data['betas'][0][None],
                body_pose = self.smpl_data['body_pose'][pose_idx][None],
                jaw_pose = self.smpl_data['jaw_pose'][pose_idx][None],
                expression = self.smpl_data['expression'][pose_idx][None],
            )

        data_item = dict()
        if self.load_smpl_pos_map:
            smpl_pos_map = cv.imread(self.data_dir + '/smpl_pos_map/%08d.exr' % pose_idx, cv.IMREAD_UNCHANGED)
            pos_map_size = smpl_pos_map.shape[1] // 2
            smpl_pos_map = np.concatenate([smpl_pos_map[:, :pos_map_size], smpl_pos_map[:, pos_map_size:]], 2)
            smpl_pos_map = smpl_pos_map.transpose((2, 0, 1))
            data_item['smpl_pos_map'] = smpl_pos_map

        if self.load_smpl_nml_map:
            smpl_nml_map = cv.imread(self.data_dir + '/smpl_nml_map/%08d.jpg' % pose_idx, cv.IMREAD_UNCHANGED)
            smpl_nml_map = (smpl_nml_map / 255.).astype(np.float32)
            nml_map_size = smpl_nml_map.shape[1] // 2
            smpl_nml_map = np.concatenate([smpl_nml_map[:, :nml_map_size], smpl_nml_map[:, nml_map_size:]], 2)
            smpl_nml_map = smpl_nml_map.transpose((2, 0, 1))
            data_item['smpl_nml_map'] = smpl_nml_map

        data_item['joints'] = live_smpl.joints[0, :22]
        data_item['kin_parent'] = self.smpl_model.parents[:22].to(torch.long)
        data_item['item_idx'] = index
        data_item['data_idx'] = data_idx
        data_item['time_stamp'] = np.array(pose_idx, np.float32)
        data_item['global_orient'] = self.smpl_data['global_orient'][pose_idx]
        data_item['transl'] = self.smpl_data['transl'][pose_idx]
        data_item['live_smpl_v'] = live_smpl.vertices[0]
        data_item['live_smpl_v_woRoot'] = live_smpl_woRoot.vertices[0]
        data_item['cano_smpl_v'] = cano_smpl.vertices[0]
        data_item['cano_jnts'] = cano_smpl.joints[0]
        data_item['cano2live_jnt_mats'] = torch.matmul(live_smpl.A[0], torch.linalg.inv(cano_smpl.A[0]))
        data_item['cano2live_jnt_mats_woRoot'] = torch.matmul(live_smpl_woRoot.A[0], torch.linalg.inv(cano_smpl.A[0]))
        data_item['cano_smpl_center'] = self.cano_smpl_center
        data_item['cano_bounds'] = self.cano_bounds
        data_item['smpl_faces'] = self.smpl_faces
        min_xyz = live_smpl.vertices[0].min(0)[0] - 0.15
        max_xyz = live_smpl.vertices[0].max(0)[0] + 0.15
        live_bounds = torch.stack([min_xyz, max_xyz], 0).to(torch.float32).numpy()
        data_item['live_bounds'] = live_bounds

        if training:
            color_img, mask_img = self.load_color_mask_images(pose_idx, view_idx)

            color_img = (color_img / 255.).astype(np.float32)

            boundary_mask_img, mask_img = self.get_boundary_mask(mask_img)

            if self.mode == '3dgs':
                data_item.update({
                    'img_h': color_img.shape[0],
                    'img_w': color_img.shape[1],
                    'extr': self.extr_mats[view_idx],
                    'intr': self.intr_mats[view_idx],
                    'color_img': color_img,
                    'mask_img': mask_img,
                    'boundary_mask_img': boundary_mask_img
                })
            elif self.mode == 'nerf':
                depth_img = np.zeros(color_img.shape[:2], np.float32)
                nerf_random = nerf_util.sample_randomly_for_nerf_rendering(
                    color_img, mask_img, depth_img,
                    self.extr_mats[view_idx], self.intr_mats[view_idx],
                    live_bounds,
                    unsample_region_mask = boundary_mask_img
                )
                data_item.update({
                    'nerf_random': nerf_random,
                    'extr': self.extr_mats[view_idx],
                    'intr': self.intr_mats[view_idx]
                })
            else:
                raise ValueError('Invalid dataset mode!')
        else:
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

        if self.mode == 'nerf' or self.mode == '3dgs' and not training:
            # mano
            data_item['left_cano_mano_v'], data_item['left_cano_mano_n'], data_item['right_cano_mano_v'], data_item['right_cano_mano_n'] \
                = commons.generate_two_manos(self, self.cano_smpl['vertices'])
            data_item['left_live_mano_v'], data_item['left_live_mano_n'], data_item['right_live_mano_v'], data_item['right_live_mano_n'] \
                = commons.generate_two_manos(self, live_smpl.vertices[0])

        return data_item

    def load_cam_data(self):
        """
        Initialize:
        self.cam_names, self.view_num, self.extr_mats, self.intr_mats,
        self.img_widths, self.img_heights
        """
        raise NotImplementedError

    def load_smpl_data(self):
        """
        Initialize:
        self.cam_data, a dict including ['body_pose', 'global_orient', 'transl', 'betas', ...]
        """
        smpl_data = np.load(self.data_dir + '/smpl_params.npz', allow_pickle = True)
        smpl_data = dict(smpl_data)
        self.smpl_data = {k: torch.from_numpy(v.astype(np.float32)) for k, v in smpl_data.items()}

    def filter_missing_files(self):
        pass

    def load_color_mask_images(self, pose_idx, view_idx):
        raise NotImplementedError

    @staticmethod
    def get_boundary_mask(mask, kernel_size = 5):
        """
        :param mask: np.uint8
        :param kernel_size:
        :return:
        """
        mask_bk = mask.copy()
        thres = 128
        mask[mask < thres] = 0
        mask[mask > thres] = 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_erode = cv.erode(mask.copy(), kernel)
        mask_dilate = cv.dilate(mask.copy(), kernel)
        boundary_mask = (mask_dilate - mask_erode) == 1
        boundary_mask = np.logical_or(boundary_mask,
                                      np.logical_and(mask_bk > 5, mask_bk < 250))

        # boundary_mask_resized = cv.resize(boundary_mask.astype(np.uint8), (0, 0), fx = 0.5, fy = 0.5)
        # cv.imshow('boundary_mask', boundary_mask_resized.astype(np.uint8) * 255)
        # cv.waitKey(0)

        return boundary_mask, mask == 1

    def compute_pca(self, n_components = 10):
        from sklearn.decomposition import PCA
        from tqdm import tqdm
        import joblib

        if not os.path.exists(self.data_dir + '/smpl_pos_map/pca_%d.ckpt' % n_components):
            pose_conds = []
            mask = None
            for pose_idx in tqdm(self.pose_list, desc = 'Loading position maps...'):
                pose_map = cv.imread(self.data_dir + '/smpl_pos_map/%08d.exr' % pose_idx, cv.IMREAD_UNCHANGED)
                pose_map = pose_map[:, :pose_map.shape[1] // 2]
                if mask is None:
                    mask = np.linalg.norm(pose_map, axis = -1) > 1e-6
                pose_conds.append(pose_map[mask])
            pose_conds = np.stack(pose_conds, 0)
            pose_conds = pose_conds.reshape(pose_conds.shape[0], -1)
            self.pca = PCA(n_components = n_components)
            self.pca.fit(pose_conds)
            joblib.dump(self.pca, self.data_dir + '/smpl_pos_map/pca_%d.ckpt' % n_components)
            self.pos_map_mask = mask
        else:
            self.pca = joblib.load(self.data_dir + '/smpl_pos_map/pca_%d.ckpt' % n_components)
            pose_map = cv.imread(sorted(glob.glob(self.data_dir + '/smpl_pos_map/0*.exr'))[0], cv.IMREAD_UNCHANGED)
            pose_map = pose_map[:, :pose_map.shape[1] // 2]
            self.pos_map_mask = np.linalg.norm(pose_map, axis = -1) > 1e-6

    def transform_pca(self, pose_conds, sigma_pca = 2.):
        pose_conds = pose_conds.reshape(1, -1)
        lowdim_pose_conds = self.pca.transform(pose_conds)
        std = np.sqrt(self.pca.explained_variance_)
        lowdim_pose_conds = np.maximum(lowdim_pose_conds, -sigma_pca * std)
        lowdim_pose_conds = np.minimum(lowdim_pose_conds, sigma_pca * std)
        new_pose_conds = self.pca.inverse_transform(lowdim_pose_conds)
        new_pose_conds = new_pose_conds.reshape(-1, 3)
        return new_pose_conds


class MvRgbDatasetTHuman4(MvRgbDatasetBase):
    def __init__(
        self,
        data_dir,
        frame_range = None,
        used_cam_ids = None,
        training = True,
        subject_name = None,
        load_smpl_pos_map = False,
        load_smpl_nml_map = False,
        mode = '3dgs'
    ):
        super(MvRgbDatasetTHuman4, self).__init__(
            data_dir,
            frame_range,
            used_cam_ids,
            training,
            subject_name,
            load_smpl_pos_map,
            load_smpl_nml_map,
            mode
        )

    def load_cam_data(self):
        import json
        cam_data = json.load(open(self.data_dir + '/calibration.json', 'r'))
        self.view_num = len(cam_data)
        self.extr_mats = []
        self.cam_names = ['cam%02d' % view_idx for view_idx in range(self.view_num)]
        for view_idx in range(self.view_num):
            extr_mat = np.identity(4, np.float32)
            extr_mat[:3, :3] = np.array(cam_data['cam%02d' % view_idx]['R'], np.float32).reshape(3, 3)
            extr_mat[:3, 3] = np.array(cam_data['cam%02d' % view_idx]['T'], np.float32)
            self.extr_mats.append(extr_mat)
        self.intr_mats = [np.array(cam_data['cam%02d' % view_idx]['K'], np.float32).reshape(3, 3) for view_idx in range(self.view_num)]
        self.img_heights = [cam_data['cam%02d' % view_idx]['imgSize'][1] for view_idx in range(self.view_num)]
        self.img_widths = [cam_data['cam%02d' % view_idx]['imgSize'][0] for view_idx in range(self.view_num)]

    def filter_missing_files(self):
        missing_data_list = []
        with open(self.data_dir + '/missing_img_files.txt', 'r') as fp:
            lines = fp.readlines()
        for line in lines:
            line = line.replace('\\', '/')  # considering both Windows and Ubuntu file system
            frame_idx = int(os.path.basename(line).replace('.jpg', ''))
            view_idx = int(os.path.basename(os.path.dirname(line)).replace('cam', ''))
            missing_data_list.append((frame_idx, view_idx))
        for missing_data_idx in missing_data_list:
            if missing_data_idx in self.data_list:
                self.data_list.remove(missing_data_idx)

    def load_color_mask_images(self, pose_idx, view_idx):
        color_img = cv.imread(self.data_dir + '/images/cam%02d/%08d.jpg' % (view_idx, pose_idx), cv.IMREAD_UNCHANGED)
        mask_img = cv.imread(self.data_dir + '/masks/cam%02d/%08d.jpg' % (view_idx, pose_idx), cv.IMREAD_UNCHANGED)
        return color_img, mask_img


class MvRgbDatasetAvatarReX(MvRgbDatasetBase):
    def __init__(
        self,
        data_dir,
        frame_range = None,
        used_cam_ids = None,
        training = True,
        subject_name = None,
        load_smpl_pos_map = False,
        load_smpl_nml_map = False,
        mode = '3dgs'
    ):
        super(MvRgbDatasetAvatarReX, self).__init__(
            data_dir,
            frame_range,
            used_cam_ids,
            training,
            subject_name,
            load_smpl_pos_map,
            load_smpl_nml_map,
            mode
        )

    def load_cam_data(self):
        import json
        cam_data = json.load(open(self.data_dir + '/calibration_full.json', 'r'))
        self.cam_names = list(cam_data.keys())
        self.view_num = len(self.cam_names)
        self.extr_mats = []
        for view_idx in range(self.view_num):
            extr_mat = np.identity(4, np.float32)
            extr_mat[:3, :3] = np.array(cam_data[self.cam_names[view_idx]]['R'], np.float32).reshape(3, 3)
            extr_mat[:3, 3] = np.array(cam_data[self.cam_names[view_idx]]['T'], np.float32)
            self.extr_mats.append(extr_mat)
        self.intr_mats = [np.array(cam_data[self.cam_names[view_idx]]['K'], np.float32).reshape(3, 3) for view_idx in range(self.view_num)]
        self.img_heights = [cam_data[self.cam_names[view_idx]]['imgSize'][1] for view_idx in range(self.view_num)]
        self.img_widths = [cam_data[self.cam_names[view_idx]]['imgSize'][0] for view_idx in range(self.view_num)]

    def filter_missing_files(self):
        if os.path.exists(self.data_dir + '/missing_img_files.txt'):
            missing_data_list = []
            with open(self.data_dir + '/missing_img_files.txt', 'r') as fp:
                lines = fp.readlines()
            for line in lines:
                line = line.replace('\\', '/')  # considering both Windows and Ubuntu file system
                frame_idx = int(os.path.basename(line).replace('.jpg', ''))
                view_idx = self.cam_names.index(os.path.basename(os.path.dirname(line)))
                missing_data_list.append((frame_idx, view_idx))
            for missing_data_idx in missing_data_list:
                if missing_data_idx in self.data_list:
                    self.data_list.remove(missing_data_idx)

    def load_color_mask_images(self, pose_idx, view_idx):
        cam_name = self.cam_names[view_idx]
        color_img = cv.imread(self.data_dir + '/%s/%08d.jpg' % (cam_name, pose_idx), cv.IMREAD_UNCHANGED)
        mask_img = cv.imread(self.data_dir + '/%s/mask/pha/%08d.jpg' % (cam_name, pose_idx), cv.IMREAD_UNCHANGED)
        return color_img, mask_img


class MvRgbDatasetActorsHQ(MvRgbDatasetBase):
    def __init__(
        self,
        data_dir,
        frame_range = None,
        used_cam_ids = None,
        training = True,
        subject_name = None,
        load_smpl_pos_map = False,
        load_smpl_nml_map = False,
        mode = '3dgs'
    ):
        super(MvRgbDatasetActorsHQ, self).__init__(
            data_dir,
            frame_range,
            used_cam_ids,
            training,
            subject_name,
            load_smpl_pos_map,
            load_smpl_nml_map,
            mode
        )

        if subject_name is None:
            self.subject_name = os.path.basename(os.path.dirname(self.data_dir))

    def load_cam_data(self):
        import csv
        cam_names = []
        extr_mats = []
        intr_mats = []
        img_widths = []
        img_heights = []
        with open(self.data_dir + '/4x/calibration.csv', "r", newline = "", encoding = 'utf-8') as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                cam_names.append(row['name'])
                img_widths.append(int(row['w']))
                img_heights.append(int(row['h']))

                extr_mat = np.identity(4, np.float32)
                extr_mat[:3, :3] = cv.Rodrigues(np.array([float(row['rx']), float(row['ry']), float(row['rz'])], np.float32))[0]
                extr_mat[:3, 3] = np.array([float(row['tx']), float(row['ty']), float(row['tz'])])
                extr_mat = np.linalg.inv(extr_mat)
                extr_mats.append(extr_mat)

                intr_mat = np.identity(3, np.float32)
                intr_mat[0, 0] = float(row['fx']) * float(row['w'])
                intr_mat[0, 2] = float(row['px']) * float(row['w'])
                intr_mat[1, 1] = float(row['fy']) * float(row['h'])
                intr_mat[1, 2] = float(row['py']) * float(row['h'])
                intr_mats.append(intr_mat)

        self.cam_names, self.img_widths, self.img_heights, self.extr_mats, self.intr_mats \
            = cam_names, img_widths, img_heights, extr_mats, intr_mats

    def load_color_mask_images(self, pose_idx, view_idx):
        cam_name = self.cam_names[view_idx]
        color_img = cv.imread(self.data_dir + '/4x/rgbs/%s/%s_rgb%06d.jpg' % (cam_name, cam_name, pose_idx), cv.IMREAD_UNCHANGED)
        mask_img = cv.imread(self.data_dir + '/4x/masks/%s/%s_mask%06d.png' % (cam_name, cam_name, pose_idx), cv.IMREAD_UNCHANGED)
        return color_img, mask_img
