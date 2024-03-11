import torch
import torch.nn as nn
import math

# from lib.modules.stylegan2 import Generator
# from lib.ops.styleGAN import grid_sample_gradfix


def conv3x3(in_channels, out_channels, stride=1, use_bn=False):
    assert stride == 1 or stride == 2
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
              nn.LeakyReLU(0.2, inplace=True)]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


def deconv3x3(in_channels, out_channels, use_bn=False):
    layers = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
              nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
              nn.LeakyReLU(0.2, inplace=True)]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


class ConvStack(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 hid_dim=None,
                 kernel_size=5,
                 layer_num=3,
                 use_relu=False,
                 ):
        super().__init__()
        assert kernel_size in [3, 5, 7]
        if hid_dim is None:
            hid_dim = out_dim
        padding = (kernel_size - 1) // 2

        layers = []
        layers.append(nn.Conv2d(in_dim, hid_dim, kernel_size=kernel_size, stride=1, padding=padding, bias=False))
        if use_relu:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        for i in range(layer_num - 2):
            layers.append(nn.Conv2d(hid_dim, hid_dim, kernel_size=kernel_size, stride=1, padding=padding, bias=False))
            if use_relu:
                layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Conv2d(hid_dim, out_dim, kernel_size=kernel_size, stride=1, padding=padding, bias=False))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Unet5d(nn.Module):
    def __init__(self, in_c, out_c, nf):
        super().__init__()
        self.conv1 = conv3x3(in_c, nf, stride=1, use_bn=False)
        self.conv2 = conv3x3(nf, 2 * nf, stride=2, use_bn=True)
        self.conv3 = conv3x3(2 * nf, 4 * nf, stride=2, use_bn=True)
        self.conv4 = conv3x3(4 * nf, 8 * nf, stride=2, use_bn=True)
        self.conv5 = conv3x3(8 * nf, 8 * nf, stride=2, use_bn=True)
        self.deconv1 = deconv3x3(8 * nf, 8 * nf, use_bn=True)
        self.deconv2 = deconv3x3(2 * 8 * nf, 4 * nf, use_bn=True)
        self.deconv3 = deconv3x3(2 * 4 * nf, 2 * nf, use_bn=True)
        self.deconv4 = deconv3x3(2 * 2 * nf, nf, use_bn=True)
        self.deconv5 = conv3x3(2 * nf, nf, stride=1, use_bn=False)
        self.tail = nn.Conv2d(nf, out_c, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        # x: bs x in_c x 128 x 128
        x1 = self.conv1(x)  # bs x nf x 128 x 128
        x2 = self.conv2(x1)  # bs x 2nf x 64 x 64
        x3 = self.conv3(x2)  # bs x 4nf x 32 x 32
        x4 = self.conv4(x3)  # bs x 8nf x 16 x 16
        x5 = self.conv5(x4)  # bs x 8nf x 8 x 8

        y1 = self.deconv1(x5)  # bs x 8nf x 16 x 16
        y2 = self.deconv2(torch.cat([y1, x4], dim=1))  # bs x 4nf x 32 x 32
        y3 = self.deconv3(torch.cat([y2, x3], dim=1))  # bs x 2nf x 64 x 64
        y4 = self.deconv4(torch.cat([y3, x2], dim=1))  # bs x nf x 128 x 128
        y5 = self.deconv5(torch.cat([y4, x1], dim=1))  # bs x nf x 128 x 128

        out = self.tail(y5)
        return out


def grid_sample(image, p2d):
    # p2d: B x ... x 2
    # image: B x C x IH x IW
    B, C, IH, IW = image.shape
    image = image.view(B, C, IH * IW)
    assert p2d.shape[0] == B
    assert p2d.shape[-1] == 2
    points_shape = list(p2d.shape[1:-1])
    p2d = p2d.contiguous().view(B, 1, -1, 2)  # B x 1 x N x 2

    ix = p2d[..., 0]  # B x 1 x N
    iy = p2d[..., 1]  # B x 1 x N
    ix = ((ix + 1) / 2) * (IW - 1)
    iy = ((iy + 1) / 2) * (IH - 1)
    with torch.no_grad():
        ix_nw = torch.floor(ix)
        iy_nw = torch.floor(iy)
        ix_ne = ix_nw + 1
        iy_ne = iy_nw
        ix_sw = ix_nw
        iy_sw = iy_nw + 1
        ix_se = ix_nw + 1
        iy_se = iy_nw + 1

    nw = (ix_se - ix) * (iy_se - iy)
    ne = (ix - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix) * (iy - iy_ne)
    se = (ix - ix_nw) * (iy - iy_nw)

    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW - 1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH - 1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW - 1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH - 1, out=iy_ne)

        torch.clamp(ix_sw, 0, IW - 1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH - 1, out=iy_sw)

        torch.clamp(ix_se, 0, IW - 1, out=ix_se)
        torch.clamp(iy_se, 0, IH - 1, out=iy_se)

    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(B, 1, -1).expand(-1, C, -1))  # B x C x N
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(B, 1, -1).expand(-1, C, -1))  # B x C x N
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(B, 1, -1).expand(-1, C, -1))  # B x C x N
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(B, 1, -1).expand(-1, C, -1))  # B x C x N

    out_val = nw_val * nw + ne_val * ne + sw_val * sw + se_val * se  # B x C x N
    out_val = out_val.permute(0, 2, 1).contiguous().view([B] + points_shape + [C])

    return out_val


def triplane_sample(xyz, fmap):
    C = fmap.shape[1] // 3
    assert fmap.shape[1] == 3 * C
    fmap_list = fmap.split(C, dim=1)
    output = []
    for fmapIdx, axisIdx1, axisIdx2 in zip([0, 1, 2], [0, 1, 2], [1, 2, 0]):
        feat = grid_sample(torch.stack([xyz[..., axisIdx1], xyz[..., axisIdx2]], dim=-1),
                           fmap_list[fmapIdx].expand(xyz.shape[0], -1, -1, -1))
        output.append(feat)
    return torch.cat(output, dim=-1)


class TriPlaneFeature(nn.Module):
    def __init__(self, feat_dim, feat_size):
        super().__init__()
        self.feat_dim = feat_dim
        self.famp = nn.Parameter(torch.randn(1, 3 * feat_dim, feat_size, feat_size).float() * 0.03)

    def forward(self, input):
        return self.famp.expand(input.shape[0], -1, -1, -1)

    @staticmethod
    def sample_feat(xyz, fmap):
        triplane_sample(xyz, fmap)


class UVFeature(nn.Module):
    def __init__(self, feat_dim, feat_size):
        super().__init__()
        self.feat_dim = feat_dim
        self.famp = nn.Parameter(torch.randn(1, feat_dim, feat_size, feat_size).float() * 0.03)

    def forward(self, input):
        return self.famp.expand(input.shape[0], -1, -1, -1)

    @staticmethod
    def sample_feat(p2d, fmap):
        return grid_sample(p2d, fmap)


# class TriPlaneFeature_StyleGAN(nn.Module):
#     def __init__(self, feat_dim, feat_size, semantic_dim=0, style_dim=512, n_mlp=8):
#         super().__init__()
#         assert 2 ** int(math.log(feat_size, 2)) == feat_size
#         self.semantic_dim = max(semantic_dim, 0)
#         self.style_dim = style_dim
#         self.feat_dim = feat_dim
#         self.fc = nn.Linear(style_dim + semantic_dim, style_dim)
#         self.generator = Generator(size=feat_size, dim=feat_dim * 3, style_dim=style_dim, n_mlp=n_mlp)

#     def forward(self, styles, semantic=None, randomize_noise=True):
#         if isinstance(styles, (list, tuple)):
#             if semantic is None:
#                 x = styles
#             else:
#                 x = [self.fc(torch.cat([s, semantic], dim=-1)) for s in styles]
#         elif isinstance(styles, torch.Tensor):
#             if semantic is None:
#                 x = [styles]
#             else:
#                 x = [torch.cat([styles, semantic], dim=-1)]
#         else:
#             raise NotImplementedError

#         fmap_x, fmap_y, fmap_z = self.generator(styles=x, randomize_noise=randomize_noise)[0].split(self.feat_dim, dim=1)
#         return [fmap_x, fmap_y, fmap_z]

#     @staticmethod
#     def sample_feat(xyz, fmap_list):
#         # xyz: B x N x 3 (-1 ~ 1)
#         # im_feat: B x C x H x W
#         # output: B x N x f
#         assert xyz.shape[-1] == 3
#         output = []
#         for fmapIdx, axisIdx1, axisIdx2 in zip([0, 1, 2], [1, 2, 0], [2, 0, 1]):
#             p2d = torch.stack([xyz[..., axisIdx1], xyz[..., axisIdx2]], dim=-1)
#             fmap = fmap_list[fmapIdx].expand(xyz.shape[0], -1, -1, -1)
#             p2d = p2d + 1.0 / fmap.shape[-1]
#             feat = grid_sample_gradfix.grid_sample(fmap, p2d.unsqueeze(2))[..., 0]
#             feat = feat.permute(0, 2, 1)
#             output.append(feat)
#         return torch.cat(output, dim=-1)


# class UVFeature_StyleGAN(nn.Module):
#     def __init__(self, feat_dim, feat_size, semantic_dim=0, style_dim=512, n_mlp=8):
#         super().__init__()
#         assert 2 ** int(math.log(feat_size, 2)) == feat_size
#         self.semantic_dim = max(semantic_dim, 0)
#         self.style_dim = style_dim
#         self.feat_dim = feat_dim
#         self.fc = nn.Linear(style_dim + semantic_dim, style_dim)
#         self.generator = Generator(size=feat_size, dim=feat_dim, style_dim=style_dim, n_mlp=n_mlp)

#     def forward(self, styles, semantic=None, randomize_noise=True):
#         if isinstance(styles, (list, tuple)):
#             if semantic is None:
#                 x = styles
#             else:
#                 x = [self.fc(torch.cat([s, semantic], dim=-1)) for s in styles]
#         elif isinstance(styles, torch.Tensor):
#             if semantic is None:
#                 x = [styles]
#             else:
#                 x = [torch.cat([styles, semantic], dim=-1)]
#         else:
#             raise NotImplementedError

#         fmap = self.generator(styles=x, randomize_noise=randomize_noise)[0]
#         return fmap

#     @staticmethod
#     def sample_feat(p2d, fmap):
#         # p2d: B x N x 2 (-1 ~ 1)
#         # im_feat: B x C x H x W
#         # output: B x N x f
#         assert p2d.shape[-1] == 2
#         fmap = fmap.expand(p2d.shape[0], -1, -1, -1)
#         p2d = p2d + 1.0 / fmap.shape[-1]
#         feat = grid_sample_gradfix.grid_sample(fmap, p2d.unsqueeze(2))[..., 0]
#         feat = feat.permute(0, 2, 1)
#         return feat
