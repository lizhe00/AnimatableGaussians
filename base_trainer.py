import os
import platform
import time
import yaml
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data
import numpy as np
import glob
import shutil

from utils.net_util import to_cuda


def worker_init_fn(worker_id):  # set numpy's random seed
    seed = torch.initial_seed()
    seed = seed % (2 ** 32)
    np.random.seed(seed + worker_id)


class BaseTrainer:
    def __init__(self, opt):
        self.opt = opt

        self.dataset = None
        self.network = None
        self.net_dict = {}
        self.optm_dict = {}
        self.update_keys = None
        self.lr_schedule_dict = {}
        self.iter_idx = 0
        self.epoch_idx = 0
        self.iter_num = 9999999999

        self.loss_weight = self.opt['train']['loss_weight']

    @staticmethod
    def load_pretrained(path, dict_):
        data = torch.load(path)
        for k in dict_:
            if k in data:
                print('# Loading %s...' % k)
                dict_[k].load_state_dict(data[k])
            else:
                print('# %s not found!' % k)
        return data.get('epoch_idx', None)

    def load_ckpt(self, path, load_optm = True):
        epoch_idx = self.load_pretrained(path + '/net.pt', self.net_dict)
        if load_optm:
            if os.path.exists(path + '/optm.pt'):
                self.load_pretrained(path + '/optm.pt', self.optm_dict)
            else:
                print('# Optimizer not found!')
        return epoch_idx

    # @staticmethod
    def save_trained(self, path, dict_):
        data = {}
        for k in dict_:
            data[k] = dict_[k].state_dict()
        data.update({
            'epoch_idx': self.epoch_idx,
        })
        torch.save(data, path)

    def save_ckpt(self, path, save_optm = True):
        self.save_trained(path + '/net.pt', self.net_dict)
        if save_optm:
            self.save_trained(path + '/optm.pt', self.optm_dict)

    def zero_grad(self):
        if self.update_keys is None:
            update_keys = self.optm_dict.keys()
        else:
            update_keys = self.update_keys
        for k in update_keys:
            self.optm_dict[k].zero_grad()

    def step(self):
        if self.update_keys is None:
            update_keys = self.optm_dict.keys()
        else:
            update_keys = self.update_keys
        for k in update_keys:
            self.optm_dict[k].step()

    def update_lr(self, iter_idx):
        lr_dict = {}
        if self.update_keys is None:
            update_keys = self.optm_dict.keys()
        else:
            update_keys = self.update_keys
        for k in update_keys:
            lr = self.lr_schedule_dict[k].get_learning_rate(iter_idx)
            for param_group in self.optm_dict[k].param_groups:
                param_group['lr'] = lr
            lr_dict[k] = lr
        return lr_dict

    def set_dataset(self, dataset):
        self.dataset = dataset

    def set_network(self, network):
        self.network = network

    def set_net_dict(self, net_dict):
        self.net_dict = net_dict

    def set_optm_dict(self, optm_dict):
        self.optm_dict = optm_dict

    def set_update_keys(self, update_keys):
        self.update_keys = update_keys

    def set_lr_schedule_dict(self, lr_schedule_dict):
        self.lr_schedule_dict = lr_schedule_dict

    def set_train(self, flag = True):
        if flag:
            for k, net in self.net_dict.items():
                if k in self.update_keys:
                    net.train()
                else:
                    net.eval()
        else:
            for k, net in self.net_dict.items():
                net.eval()

    def train(self):
        # log
        os.makedirs(self.opt['train']['net_ckpt_dir'], exist_ok = True)
        log_dir = self.opt['train']['net_ckpt_dir'] + '/' + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        os.makedirs(log_dir, exist_ok = True)
        writer = SummaryWriter(log_dir)
        yaml.dump(self.opt, open(log_dir + '/config_bk.yaml', 'w'), sort_keys = False)

        self.set_train()
        self.dataset.training = True
        batch_size = self.opt['train'].get('batch_size', 1)
        num_workers = self.opt['train'].get('num_workers', 0)
        dataloader = torch.utils.data.DataLoader(self.dataset,
                                                 batch_size = batch_size,
                                                 shuffle = True,
                                                 num_workers = num_workers,
                                                 worker_init_fn = worker_init_fn,
                                                 drop_last = True)
        self.batch_num = len(self.dataset) // batch_size

        if self.opt['train'].get('save_init_ckpt', False) and self.opt['train'].get('start_epoch', 0) == 0:
            init_folder = self.opt['train']['net_ckpt_dir'] + '/init_ckpt'
            if not os.path.exists(init_folder) or self.opt['train']['start_epoch'] == 0:
                os.makedirs(init_folder, exist_ok = True)
                self.save_ckpt(init_folder, False)
            else:
                print('# Init checkpoint has been saved!')

        if self.opt['train']['prev_ckpt'] is not None:
            start_epoch = self.load_ckpt(self.opt['train']['prev_ckpt']) + 1
        else:
            prev_ckpt_path = self.opt['train']['net_ckpt_dir'] + '/epoch_latest'
            if os.path.exists(prev_ckpt_path):
                start_epoch = self.load_ckpt(prev_ckpt_path) + 1
            else:
                start_epoch = None

        if start_epoch is None:
            start_epoch = self.opt['train'].get('start_epoch', 0)
        end_epoch = self.opt['train'].get('end_epoch', 999)

        forward_one_pass = self.forward_one_pass

        for epoch_idx in range(start_epoch, end_epoch):
            self.epoch_idx = epoch_idx
            self.update_config_before_epoch(epoch_idx)
            epoch_losses = dict()

            time0 = time.time()
            for batch_idx, items in enumerate(dataloader):
                iter_idx = batch_idx + self.batch_num * epoch_idx
                self.iter_idx = iter_idx
                lr_dict = self.update_lr(iter_idx)
                items = to_cuda(items)

                loss, batch_losses = forward_one_pass(items)
                # self.zero_grad()
                # loss.backward()
                # self.step()

                # record batch loss
                log_info = 'epoch %d, batch %d, ' % (epoch_idx, batch_idx)
                log_info += 'lr: '
                for k in lr_dict.keys():
                    log_info += '%s %e, ' % (k, lr_dict[k])
                for key in batch_losses.keys():
                    log_info = log_info + ('%s: %f, ' % (key, batch_losses[key]))
                    writer.add_scalar('%s/Batch' % key, batch_losses[key], iter_idx)
                    if key in epoch_losses:
                        epoch_losses[key] += batch_losses[key]
                    else:
                        epoch_losses[key] = batch_losses[key]
                print(log_info)

                with open(os.path.join(log_dir, 'loss.txt'), 'a') as fp:
                    # record loss weight
                    if batch_idx == 0:
                        loss_weights_info = ''
                        for k in self.opt['train']['loss_weight'].keys():
                            loss_weights_info += '%s: %f, ' % (k, self.opt['train']['loss_weight'][k])
                        fp.write('# Loss weights: \n' + loss_weights_info + '\n')
                    fp.write(log_info + '\n')

                if iter_idx % self.opt['train']['ckpt_interval']['batch'] == 0 and iter_idx != 0:
                    for folder in glob.glob(self.opt['train']['net_ckpt_dir'] + '/batch_*'):
                        shutil.rmtree(folder)
                    model_folder = self.opt['train']['net_ckpt_dir'] + '/batch_%d' % iter_idx
                    os.makedirs(model_folder, exist_ok = True)
                    self.save_ckpt(model_folder, save_optm = False)

                if iter_idx % self.opt['train']['eval_interval'] == 0 and iter_idx != 0:
                # if True:
                    self.mini_test()
                    self.set_train()
                time1 = time.time()
                print('One iteration costs %f secs' % (time1 - time0))
                time0 = time1

                if iter_idx == self.iter_num:
                    return

            """ EPOCH """
            # record epoch loss
            for key in epoch_losses.keys():
                epoch_losses[key] /= self.batch_num
                writer.add_scalar('%s/Epoch' % key, epoch_losses[key], epoch_idx)

            if epoch_idx % self.opt['train']['ckpt_interval']['epoch'] == 0:
                model_folder = self.opt['train']['net_ckpt_dir'] + '/epoch_%d' % epoch_idx
                os.makedirs(model_folder, exist_ok = True)
                self.save_ckpt(model_folder)

            if self.batch_num > 50:
                latest_folder = self.opt['train']['net_ckpt_dir'] + '/epoch_latest'
                os.makedirs(latest_folder, exist_ok = True)
                self.save_ckpt(latest_folder)
        writer.close()

    @torch.no_grad()
    def mini_test(self):
        """ Test during training """
        pass

    def forward_one_pass(self, items):
        raise NotImplementedError('"forward_one_pass" method is not implemented!')

    def update_config_before_epoch(self, epoch_idx):
        pass
