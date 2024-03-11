import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


def init_out_weights(self):
    for m in self.modules():
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param.data, -1e-5, 1e-5)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0.0)


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels = [512, 512, 512, 343, 512, 512],
                 res_layers = [], nlactv = nn.ReLU(), last_op=None, norm = None, init_last_layer = False):
        super(MLP, self).__init__()

        self.nlactv = nlactv

        self.fc_list = nn.ModuleList()
        self.res_layers = res_layers
        if self.res_layers is None:
            self.res_layers = []

        self.all_channels = [in_channels] + inter_channels + [out_channels]
        for l in range(0, len(self.all_channels) - 2):
            if l in self.res_layers:
                if norm == 'weight':
                    # print('layer %d weight normalization in fusion mlp' % l)
                    self.fc_list.append(nn.Sequential(
                        nn.utils.weight_norm(nn.Conv1d(self.all_channels[l] + self.all_channels[0], self.all_channels[l + 1], 1)),
                        self.nlactv
                    ))
                else:
                    self.fc_list.append(nn.Sequential(
                        nn.Conv1d(self.all_channels[l] + self.all_channels[0], self.all_channels[l + 1], 1),
                        self.nlactv
                    ))
                self.all_channels[l] += self.all_channels[0]
            else:
                if norm == 'weight':
                    # print('layer %d weight normalization in fusion mlp' % l)
                    self.fc_list.append(nn.Sequential(
                        nn.utils.weight_norm(nn.Conv1d(self.all_channels[l], self.all_channels[l + 1], 1)),
                        self.nlactv
                    ))
                else:
                    self.fc_list.append(nn.Sequential(
                        nn.Conv1d(self.all_channels[l], self.all_channels[l + 1], 1),
                        self.nlactv
                    ))

        self.fc_list.append(nn.Conv1d(self.all_channels[-2], out_channels, 1))

        if init_last_layer:
            self.fc_list[-1].apply(init_out_weights)

        if last_op == 'sigmoid':
            self.last_op = nn.Sigmoid()
        elif last_op == 'tanh':
            self.last_op = nn.Tanh()
        else:
            self.last_op = None

    def forward(self, x, return_inter_layer = []):
        tmpx = x
        inter_feat_list = []
        for i, fc in enumerate(self.fc_list):
            if i in self.res_layers:
                x = fc(torch.cat([x, tmpx], dim = 1))
            else:
                x = fc(x)
            if i == len(self.fc_list) - 1 and self.last_op is not None:  # last layer
                x = self.last_op(x)
            if i in return_inter_layer:
                inter_feat_list.append(x.clone())

        if len(return_inter_layer) > 0:
            return x, inter_feat_list
        else:
            return x


class MLPLinear(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 inter_channels,
                 res_layers = [],
                 nlactv = nn.ReLU(),
                 last_op = None):
        super(MLPLinear, self).__init__()

        self.fc_list = nn.ModuleList()
        self.all_channels = [in_channels] + inter_channels + [out_channels]
        self.res_layers = res_layers
        self.nlactv = nlactv
        self.last_op = last_op

        for l in range(0, len(self.all_channels) - 2):
            if l in self.res_layers:
                self.all_channels[l] += in_channels
            self.fc_list.append(
                nn.Sequential(
                    nn.Linear(self.all_channels[l], self.all_channels[l + 1]),
                    self.nlactv
                )
            )
        self.fc_list.append(nn.Linear(self.all_channels[-2], self.all_channels[-1]))

    def forward(self, x):
        tmpx = x
        for i, layer in enumerate(self.fc_list):
            if i in self.res_layers:
                x = torch.cat([x, tmpx], dim = -1)
            x = layer(x)
        if self.last_op is not None:
            x = self.last_op(x)
        return x


def parallel_concat(tensors: list, n_parallel_group: int):
    """
    :param tensors: list of tensors, each of which has a shape of [B, G*C, N]
    :param n_parallel_group:
    :return: [B, G*C', N]
    """
    batch_size = tensors[0].shape[0]
    point_num = tensors[0].shape[-1]
    assert all([t.shape[0] == batch_size for t in tensors]), 'All tensors should have the same batch size'
    assert all([t.shape[2] == point_num for t in tensors]), 'All tensors should have the same point num'
    assert all([t.shape[1] % n_parallel_group==0 for t in tensors]), 'Invalid tensor channels'

    tensors_ = [
        t.reshape(batch_size, n_parallel_group, -1, point_num) for t in tensors
    ]
    concated = torch.cat(tensors_, dim=2)
    concated = concated.reshape(batch_size, -1, point_num)
    return concated


class ParallelMLP(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 group_num,
                 inter_channels,
                 res_layers = [],
                 nlactv = nn.ReLU(),
                 last_op = None):
        super(ParallelMLP, self).__init__()

        self.fc_list = nn.ModuleList()
        self.all_channels = [in_channels] + inter_channels + [out_channels]
        self.group_num = group_num
        self.res_layers = res_layers
        self.nlactv = nlactv
        self.last_op = last_op

        for l in range(0, len(self.all_channels) - 2):
            if l in self.res_layers:
                self.all_channels[l] += in_channels
            self.fc_list.append(
                nn.Sequential(
                    nn.Conv1d(self.all_channels[l] * self.group_num, self.all_channels[l + 1] * self.group_num, 1, groups = self.group_num),
                    self.nlactv
                )
            )
        self.fc_list.append(nn.Conv1d(self.all_channels[-2] * self.group_num, self.all_channels[-1] * self.group_num, 1, groups = self.group_num))

    def forward(self, x):
        """
        :param x: (batch_size, group_num, point_num, in_channels)
        :return: (batch_size, group_num, point_num, out_channels)
        """
        assert len(x.shape) == 4, 'input tensor should be a shape of [B, G, N, C]'
        assert x.shape[1] == self.group_num, 'input tensor should have %d parallel groups, but it has %s' % (self.group_num, x.shape[1])

        B, G, N, C = x.shape
        x = x.permute(0, 1, 3, 2).reshape(B, G * C, N)
        tmpx = x
        for i, layer in enumerate(self.fc_list):
            if i in self.res_layers:
                x = parallel_concat([x, tmpx], G)
            x = layer(x)
        if self.last_op is not None:
            x = self.last_op(x)
        x = x.view(B, G, -1, N).permute(0, 1, 3, 2)
        return x


class SdfMLP(MLPLinear):
    def __init__(self,
                 in_channels,
                 out_channels,
                 inter_channels,
                 res_layers = [],
                 nlactv = nn.Softplus(beta = 100),
                 geometric_init = True,
                 bias = 0.5,
                 weight_norm = True
                 ):
        super(SdfMLP, self).__init__(in_channels,
                                     out_channels,
                                     inter_channels,
                                     res_layers,
                                     nlactv,
                                     None)

        for l, layer in enumerate(self.fc_list):
            if isinstance(layer, nn.Sequential):
                lin = layer[0]
            elif isinstance(layer, nn.Linear):
                lin = layer
            else:
                raise TypeError('Invalid %d layer' % l)
            if geometric_init:
                in_dim, out_dim = lin.in_features, lin.out_features
                if l == len(self.fc_list) - 1:
                    torch.nn.init.normal_(lin.weight, mean = np.sqrt(np.pi) / np.sqrt(in_dim), std = 0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif l in self.res_layers:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(in_channels - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                if isinstance(layer, nn.Sequential):
                    layer[0] = nn.utils.weight_norm(lin)
                elif isinstance(layer, nn.Linear):
                    layer = nn.utils.weight_norm(lin)


class OffsetDecoder(nn.Module):
    """
    Same architecture with ShapeDecoder in POP (https://github.com/qianlim/POP).
    """
    def __init__(self, in_size, hsize = 256, actv_fn='softplus'):
        self.hsize = hsize
        super(OffsetDecoder, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_size, self.hsize, 1)
        self.conv2 = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv3 = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv4 = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv5 = torch.nn.Conv1d(self.hsize+in_size, self.hsize, 1)
        self.conv6 = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv7 = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv8 = torch.nn.Conv1d(self.hsize, 3, 1)
        nn.init.uniform_(self.conv8.weight, -1e-5, 1e-5)
        nn.init.constant_(self.conv8.bias, 0.)

        self.bn1 = torch.nn.BatchNorm1d(self.hsize)
        self.bn2 = torch.nn.BatchNorm1d(self.hsize)
        self.bn3 = torch.nn.BatchNorm1d(self.hsize)
        self.bn4 = torch.nn.BatchNorm1d(self.hsize)

        self.bn5 = torch.nn.BatchNorm1d(self.hsize)
        self.bn6 = torch.nn.BatchNorm1d(self.hsize)
        self.bn7 = torch.nn.BatchNorm1d(self.hsize)

        self.actv_fn = nn.ReLU() if actv_fn=='relu' else nn.Softplus()

    def forward(self, x):
        x1 = self.actv_fn(self.bn1(self.conv1(x)))
        x2 = self.actv_fn(self.bn2(self.conv2(x1)))
        x3 = self.actv_fn(self.bn3(self.conv3(x2)))
        x4 = self.actv_fn(self.bn4(self.conv4(x3)))
        x5 = self.actv_fn(self.bn5(self.conv5(torch.cat([x,x4],dim=1))))

        # position pred
        x6 = self.actv_fn(self.bn6(self.conv6(x5)))
        x7 = self.actv_fn(self.bn7(self.conv7(x6)))
        x8 = self.conv8(x7)

        return x8

    def forward_wo_bn(self, x):
        x1 = self.actv_fn(self.conv1(x))
        x2 = self.actv_fn(self.conv2(x1))
        x3 = self.actv_fn(self.conv3(x2))
        x4 = self.actv_fn(self.conv4(x3))
        x5 = self.actv_fn(self.conv5(torch.cat([x,x4],dim=1)))

        # position pred
        x6 = self.actv_fn(self.conv6(x5))
        x7 = self.actv_fn(self.conv7(x6))
        x8 = self.conv8(x7)

        return x8


class ShapeDecoder(nn.Module):
    '''
    The "Shape Decoder" in the POP paper Fig. 2. The same as the "shared MLP" in the SCALE paper.
    - with skip connection from the input features to the 4th layer's output features (like DeepSDF)
    - branches out at the second-to-last layer, one branch for position pred, one for normal pred
    '''
    def __init__(self, in_size, hsize = 256, actv_fn='softplus'):
        self.hsize = hsize
        super(ShapeDecoder, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_size, self.hsize, 1)
        self.conv2 = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv3 = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv4 = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv5 = torch.nn.Conv1d(self.hsize+in_size, self.hsize, 1)
        self.conv6 = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv7 = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv8 = torch.nn.Conv1d(self.hsize, 3, 1)

        self.conv6N = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv7N = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv8N = torch.nn.Conv1d(self.hsize, 3, 1)

        self.bn1 = torch.nn.BatchNorm1d(self.hsize)
        self.bn2 = torch.nn.BatchNorm1d(self.hsize)
        self.bn3 = torch.nn.BatchNorm1d(self.hsize)
        self.bn4 = torch.nn.BatchNorm1d(self.hsize)

        self.bn5 = torch.nn.BatchNorm1d(self.hsize)
        self.bn6 = torch.nn.BatchNorm1d(self.hsize)
        self.bn7 = torch.nn.BatchNorm1d(self.hsize)

        self.bn6N = torch.nn.BatchNorm1d(self.hsize)
        self.bn7N = torch.nn.BatchNorm1d(self.hsize)

        self.actv_fn = nn.ReLU() if actv_fn=='relu' else nn.Softplus()

        # init last layer
        nn.init.uniform_(self.conv8.weight, -1e-5, 1e-5)
        nn.init.constant_(self.conv8.bias, 0)

    def forward(self, x):
        x1 = self.actv_fn(self.bn1(self.conv1(x)))
        x2 = self.actv_fn(self.bn2(self.conv2(x1)))
        x3 = self.actv_fn(self.bn3(self.conv3(x2)))
        x4 = self.actv_fn(self.bn4(self.conv4(x3)))
        x5 = self.actv_fn(self.bn5(self.conv5(torch.cat([x,x4],dim=1))))

        # position pred
        x6 = self.actv_fn(self.bn6(self.conv6(x5)))
        x7 = self.actv_fn(self.bn7(self.conv7(x6)))
        x8 = self.conv8(x7)

        # normals pred
        xN6 = self.actv_fn(self.bn6N(self.conv6N(x5)))
        xN7 = self.actv_fn(self.bn7N(self.conv7N(xN6)))
        xN8 = self.conv8N(xN7)

        return x8, xN8


class MLPwoWeight(object):
    def __init__(self,
                 in_channels,
                 out_channels,
                 inter_channels,
                 res_layers = [],
                 nlactv = nn.ReLU(),
                 last_op = None):
        super(MLPwoWeight, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.all_channels = [in_channels] + inter_channels + [out_channels]
        self.res_layers = res_layers

        self.nlactv = nlactv
        self.last_op = last_op

        self.param_num = 0
        for i in range(len(self.all_channels) - 1):
            in_ch = self.all_channels[i]
            if i in self.res_layers:
                in_ch += self.in_channels
            out_ch = self.all_channels[i + 1]
            self.param_num += (in_ch * out_ch + out_ch)
        self.param_num_per_group = self.param_num

    def forward(self, x, params):
        """
        :param x: (batch_size, point_num, in_channels)
        :param params: (param_num, )
        :return: (batch_size, point_num, out_channels)
        """
        x = x.permute(0, 2, 1)  # (B, C, N)
        tmpx = x

        param_id = 0
        for i in range(len(self.all_channels) - 1):
            in_ch = self.all_channels[i]
            if i in self.res_layers:
                in_ch += self.in_channels
                x = torch.cat([x, tmpx], 1)
            out_ch = self.all_channels[i + 1]

            weight_len = out_ch * in_ch
            weight = params[param_id: param_id + weight_len].reshape(out_ch, in_ch, 1)
            param_id += weight_len

            bias_len = out_ch
            bias = params[param_id: param_id + bias_len]
            param_id += bias_len

            x = F.conv1d(x, weight, bias)
            if i < len(self.all_channels) - 2:
                x = self.nlactv(x)

        if self.last_op is not None:
            x = self.last_op(x)
        return x.permute(0, 2, 1)

    def __repr__(self):
        main_str = self.__class__.__name__ + '(\n'
        for i in range(len(self.all_channels) - 1):
            main_str += '\tF.conv1d(in_features=%d, out_features=%d, bias=True)\n' % (self.all_channels[i], self.all_channels[i + 1])
        main_str += '\tnlactv: %s\n' % self.nlactv.__repr__()
        main_str += ')'
        return main_str


class ParallelMLPwoWeight(object):
    def __init__(self,
                 in_channels,
                 out_channels,
                 inter_channels,
                 group_num = 1,
                 res_layers = [],
                 nlactv = nn.ReLU(),
                 last_op = None):
        super(ParallelMLPwoWeight, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.all_channels = [in_channels] + inter_channels + [out_channels]
        self.res_layers = res_layers
        self.group_num = group_num

        self.nlactv = nlactv
        self.last_op = last_op

        self.param_num = 0
        for i in range(len(self.all_channels) - 1):
            in_ch = self.all_channels[i]
            if i in self.res_layers:
                in_ch += self.in_channels
            out_ch = self.all_channels[i + 1]
            self.param_num += (in_ch * out_ch + out_ch) * self.group_num
        self.param_num_per_group = self.param_num // self.group_num

    def forward(self, x, params):
        """
        :param x: (batch_size, group_num, point_num, in_channels)
        :param params: (group_num, param_num)
        :return: (batch_size, group_num, point_num, out_channels)
        """
        batch_size, group_num, point_num, in_channels = x.shape
        assert group_num == self.group_num and in_channels == self.in_channels
        x = x.permute(0, 1, 3, 2)  # (B, G, C, N)
        x = x.reshape(batch_size, group_num * in_channels, point_num)
        tmpx = x

        param_id = 0
        for i in range(len(self.all_channels) - 1):
            in_ch = self.all_channels[i]
            if i in self.res_layers:
                in_ch += self.in_channels
                x = parallel_concat([x, tmpx], group_num)
            out_ch = self.all_channels[i + 1]

            weight_len = out_ch * in_ch
            weight = params[:, param_id: param_id + weight_len].reshape(group_num * out_ch, in_ch, 1)
            param_id += weight_len

            bias_len = out_ch
            bias = params[:, param_id: param_id + bias_len].reshape(group_num * out_ch)
            param_id += bias_len

            x = F.conv1d(x, weight, bias, groups = group_num)
            if i < len(self.all_channels) - 2:
                x = self.nlactv(x)

        if self.last_op is not None:
            x = self.last_op(x)
        x = x.reshape(batch_size, group_num, self.out_channels, point_num)
        return x.permute(0, 1, 3, 2)

    def __repr__(self):
        main_str = self.__class__.__name__ + '(\n'
        main_str += '\tgroup_num: %d\n' % self.group_num
        for i in range(len(self.all_channels) - 1):
            main_str += '\tF.conv1d(in_features=%d, out_features=%d, bias=True)\n' % (self.all_channels[i], self.all_channels[i + 1])
        main_str += '\tnlactv: %s\n' % self.nlactv.__repr__()
        main_str += ')'
        return main_str
