import os
import torch
import numpy as np
import pytorch3d.ops
import importlib

from base_trainer import BaseTrainer
import config
from network.template import TemplateNet
from network.lpips import LPIPS
import utils.lr_schedule as lr_schedule
import utils.net_util as net_util
import utils.recon_util as recon_util
from utils.net_util import to_cuda
from utils.obj_io import save_mesh_as_ply


class TemplateTrainer(BaseTrainer):
    def __init__(self, opt):
        super(TemplateTrainer, self).__init__(opt)
        self.iter_num = 15_0000

    def update_config_before_epoch(self, epoch_idx):
        self.iter_idx = epoch_idx * self.batch_num

        print('# Optimizable variable number in network: %d' % sum(p.numel() for p in self.network.parameters() if p.requires_grad))

    def forward_one_pass(self, items):
        total_loss = 0
        batch_losses = {}

        """ random sampling """
        if 'nerf_random' in items:
            items.update(items['nerf_random'])
            render_output = self.network.render(items, depth_guided_sampling = self.opt['train']['depth_guided_sampling'])

            # color loss
            if 'rgb_map' in render_output:
                color_loss = torch.nn.L1Loss()(render_output['rgb_map'], items['color_gt'])
                total_loss += self.loss_weight['color'] * color_loss
                batch_losses.update({
                    'color_loss_random': color_loss.item()
                })

            # mask loss
            if 'acc_map' in render_output:
                mask_loss = torch.nn.L1Loss()(render_output['acc_map'], items['mask_gt'])
                total_loss += self.loss_weight['mask'] * mask_loss
                batch_losses.update({
                    'mask_loss_random': mask_loss.item()
                })

            # eikonal loss
            if 'normal' in render_output:
                eikonal_loss = ((torch.linalg.norm(render_output['normal'], dim = -1) - 1.) ** 2).mean()
                total_loss += self.loss_weight['eikonal'] * eikonal_loss
                batch_losses.update({
                    'eikonal_loss': eikonal_loss.item()
                })

        self.zero_grad()
        total_loss.backward()
        self.step()

        return total_loss, batch_losses

    def run(self):
        dataset_module = self.opt['train'].get('dataset', 'MvRgbDatasetAvatarReX')
        MvRgbDataset = importlib.import_module('dataset.dataset_mv_rgb').__getattribute__(dataset_module)
        self.set_dataset(MvRgbDataset(**self.opt['train']['data']))
        self.set_network(TemplateNet(self.opt['model']).to(config.device))
        self.set_net_dict({
            'network': self.network
        })
        self.set_optm_dict({
            'network': torch.optim.Adam(self.network.parameters(), lr = 1e-3)
        })
        self.set_lr_schedule_dict({
            'network': lr_schedule.get_learning_rate_schedules(**self.opt['train']['lr']['network'])
        })
        self.set_update_keys(['network'])
        if self.opt['train'].get('finetune_hand', False):
            print('# Finetune hand')
            for n, p in self.network.named_parameters():
                if not (n.startswith('left_hand') or n.startswith('right_hand')):
                    p.requires_grad_(False)

        if 'lpips' in self.opt['train']['loss_weight']:
            self.lpips = LPIPS(net = 'vgg').to(config.device)
            for p in self.lpips.parameters():
                p.requires_grad = False

        self.train()

        # output final cano geometry
        items = to_cuda(self.dataset.getitem(0, training = False), add_batch = True)
        with torch.no_grad():
            self.network.eval()
            vertices, faces, normals = self.test_geometry(items, space = 'cano', testing_res = (256, 256, 128))
            save_mesh_as_ply(self.opt['train']['data']['data_dir'] + '/template.ply',
                             vertices, faces, normals)

    def test_geometry(self, items, space = 'live', testing_res = (128, 128, 128)):
        if space == 'live':
            bounds = items['live_bounds'][0]
        else:
            bounds = items['cano_bounds'][0]
        vol_pts = net_util.generate_volume_points(bounds, testing_res)
        chunk_size = 256 * 256 * 4
        # chunk_size = 256 * 32
        sdf_list = []
        for i in range(0, vol_pts.shape[0], chunk_size):
            vol_pts_chunk = vol_pts[i: i + chunk_size][None]
            sdf_chunk = torch.zeros(vol_pts_chunk.shape[1]).to(vol_pts_chunk)
            if space == 'live':
                cano_pts_chunk, near_flag = self.network.transform_live2cano(vol_pts_chunk, items, near_thres = 0.1)
            else:
                cano_pts_chunk = vol_pts_chunk
                dists, _, _ = pytorch3d.ops.knn_points(cano_pts_chunk, items['cano_smpl_v'], K = 1)
                near_flag = dists[:, :, 0] < (0.1**2)  # (1, N)
                near_flag.fill_(True)
                if (~near_flag).sum() > 0:
                    sdf_chunk[~near_flag[0]] = self.network.cano_weight_volume.forward_sdf(cano_pts_chunk[~near_flag][None])[0, :, 0]
            if near_flag.sum() > 0:
                ret = self.network.forward_cano_radiance_field(cano_pts_chunk[near_flag][None], None, items)
                if self.network.with_hand:
                    self.network.fuse_hands(ret, vol_pts_chunk[near_flag][None], None, items, space)
                sdf_chunk[near_flag[0]] = ret['sdf'][0, :, 0]
            # sdf_chunk = self.network.forward_cano_radiance_field(cano_pts_chunk, None, items['pose'])['sdf']
            sdf_list.append(sdf_chunk)
        sdf_list = torch.cat(sdf_list, 0)
        vertices, faces, normals = recon_util.recon_mesh(sdf_list, testing_res, bounds, iso_value = 0.)
        return vertices, faces, normals

    @torch.no_grad()
    def mini_test(self):
        self.network.eval()

        item = self.dataset.getitem(0, training = False)
        items = to_cuda(item, add_batch = True)
        vertices, faces, normals = self.test_geometry(items, space = 'cano', testing_res = (256, 256, 128))
        output_dir = self.opt['train']['net_ckpt_dir'] + '/eval'
        os.makedirs(output_dir, exist_ok = True)
        save_mesh_as_ply(output_dir + '/batch_%d.ply' % self.iter_idx, vertices, faces, normals)

        self.network.train()


if __name__ == '__main__':
    torch.manual_seed(31359)
    np.random.seed(31359)

    from argparse import ArgumentParser

    arg_parser = ArgumentParser()
    arg_parser.add_argument('-c', '--config_path', type = str, help = 'Configuration file path.')
    args = arg_parser.parse_args()

    config.load_global_opt(args.config_path)

    trainer = TemplateTrainer(config.opt)
    trainer.run()
