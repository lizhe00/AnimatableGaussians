import math
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .fused_act import FusedLeakyReLU, fused_leaky_relu
from .upfirdn2d import upfirdn2d
from . import conv2d_gradfix


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualConv2d(nn.Module):
    def __init__(
            self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = conv2d_gradfix.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class EqualLinear(nn.Module):
    def __init__(
            self, in_dim, out_dim, bias=True, bias_init=0.0, lr_mul=1.0, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


class ModulatedConv2d(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            demodulate=True,
            upsample=False,
            downsample=False,
            blur_kernel=[1, 3, 3, 1],
            fused=True,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate
        self.fused = fused

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        if not self.fused:
            weight = self.scale * self.weight.squeeze(0)
            style = self.modulation(style)
            if self.demodulate:
                w = weight.unsqueeze(0) * style.view(batch, 1, in_channel, 1, 1)
                dcoefs = (w.square().sum((2, 3, 4)) + 1e-8).rsqrt()

            input = input * style.reshape(batch, in_channel, 1, 1)

            if self.upsample:
                weight = weight.transpose(0, 1)
                out = conv2d_gradfix.conv_transpose2d(
                    input, weight, padding=0, stride=2
                )
                out = self.blur(out)

            elif self.downsample:
                input = self.blur(input)
                out = conv2d_gradfix.conv2d(input, weight, padding=0, stride=2)

            else:
                out = conv2d_gradfix.conv2d(input, weight, padding=self.padding)

            if self.demodulate:
                out = out * dcoefs.view(batch, -1, 1, 1)

            return out

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = conv2d_gradfix.conv_transpose2d(
                input, weight, padding=0, stride=2, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = conv2d_gradfix.conv2d(
                input, weight, padding=0, stride=2, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = conv2d_gradfix.conv2d(
                input, weight, padding=self.padding, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()
        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class ConvLayer(nn.Sequential):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            downsample=False,
            blur_kernel=[1, 3, 3, 1],
            bias=True,
            activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            layers.append(FusedLeakyReLU(out_channel, bias=bias))

        super().__init__(*layers)


def get_haar_wavelet(in_channels):
    haar_wav_l = 1 / (2 ** 0.5) * torch.ones(1, 2)
    haar_wav_h = 1 / (2 ** 0.5) * torch.ones(1, 2)
    haar_wav_h[0, 0] = -1 * haar_wav_h[0, 0]

    haar_wav_ll = haar_wav_l.T * haar_wav_l
    haar_wav_lh = haar_wav_h.T * haar_wav_l
    haar_wav_hl = haar_wav_l.T * haar_wav_h
    haar_wav_hh = haar_wav_h.T * haar_wav_h

    return haar_wav_ll, haar_wav_lh, haar_wav_hl, haar_wav_hh


class HaarTransform(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        ll, lh, hl, hh = get_haar_wavelet(in_channels)

        self.register_buffer('ll', ll)
        self.register_buffer('lh', lh)
        self.register_buffer('hl', hl)
        self.register_buffer('hh', hh)

    def forward(self, input):
        ll = upfirdn2d(input, self.ll, down=2)
        lh = upfirdn2d(input, self.lh, down=2)
        hl = upfirdn2d(input, self.hl, down=2)
        hh = upfirdn2d(input, self.hh, down=2)

        return torch.cat((ll, lh, hl, hh), 1)


class InverseHaarTransform(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        ll, lh, hl, hh = get_haar_wavelet(in_channels)

        self.register_buffer('ll', ll)
        self.register_buffer('lh', -lh)
        self.register_buffer('hl', -hl)
        self.register_buffer('hh', hh)

    def forward(self, input):
        ll, lh, hl, hh = input.chunk(4, 1)
        ll = upfirdn2d(ll, self.ll, up=2, pad=(1, 0, 1, 0))
        lh = upfirdn2d(lh, self.lh, up=2, pad=(1, 0, 1, 0))
        hl = upfirdn2d(hl, self.hl, up=2, pad=(1, 0, 1, 0))
        hh = upfirdn2d(hh, self.hh, up=2, pad=(1, 0, 1, 0))

        return ll + lh + hl + hh


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], downsample=True):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=downsample)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        return out


class FromRGB(nn.Module):
    def __init__(self, out_channel, in_channel, downsample=True, blur_kernel=[1, 3, 3, 1], use_wt=True):
        super().__init__()

        self.downsample = downsample
        self.use_wt = use_wt
        if downsample:
            self.downsample = Downsample(blur_kernel)
            if use_wt:
                self.iwt = InverseHaarTransform(in_channel)
                self.dwt = HaarTransform(in_channel)
        self.in_channel = in_channel * 4 if self.use_wt else in_channel
        self.conv = ConvLayer(self.in_channel, out_channel, 1)

    def forward(self, input, skip=None):
        if self.downsample:
            if self.use_wt:
                input = self.iwt(input)  # [1024, 3]
                input = self.downsample(input)  # [512, 3]
                input = self.dwt(input)  # [256, 12]
            else:
                input = self.downsample(input)  # [512, 3]

        out = self.conv(input)  # [256, out_channel]

        if skip is not None:
            out = out + skip

        return input, out


class Discriminator(nn.Module):
    def __init__(self, size, img_channel=6, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], c_dim=0):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
            2048: 16 * channel_multiplier,
            4096: 16 * channel_multiplier
        }

        self.dwt = HaarTransform(img_channel)

        self.from_rgbs = nn.ModuleList()
        self.convs = nn.ModuleList()

        log_size = int(math.log(size, 2)) - 1

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            self.from_rgbs.append(FromRGB(in_channel, img_channel, downsample=i != log_size))
            self.convs.append(ConvBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.from_rgbs.append(FromRGB(channels[4], img_channel))

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

        self.c_dim = c_dim
        if c_dim > 0:
            style_dim = 64
            lr_mlp = 0.01
            layers = []
            layers.append(
                EqualLinear(
                    c_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
                )
            )
            for i in range(3):
                layers.append(
                    EqualLinear(
                        style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
                    )
                )
            self.mapping = nn.Sequential(*layers)

    def forward(self, input, flat_pose=None):
        input = self.dwt(input)
        out = None

        for from_rgb, conv in zip(self.from_rgbs, self.convs):
            input, out = from_rgb(input, out)
            out = conv(out)

        _, out = self.from_rgbs[-1](input, out)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(group, -1, self.stddev_feat, channel // self.stddev_feat, height, width)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)
        if self.c_dim > 0:
            pose_embed = self.mapping(flat_pose)
            pose_embed = self.normalize_2nd_moment(pose_embed)
            out = (out * pose_embed).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.c_dim))

        return out

    def normalize_2nd_moment(self, x, dim=1, eps=1e-8):
        return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


class StyledConv(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=False,
            blur_kernel=[1, 3, 3, 1],
            demodulate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, out_channel=12, upsample=True, blur_kernel=[1, 3, 3, 1], use_wt=True):
        super().__init__()
        self.use_wt = use_wt
        if upsample:
            self.upsample = Upsample(blur_kernel)
            if use_wt:
                self.iwt = InverseHaarTransform(3)
                self.dwt = HaarTransform(3)
        self.out_channel = out_channel if self.use_wt else out_channel // 4
        self.conv = ModulatedConv2d(in_channel, self.out_channel, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, self.out_channel, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            if self.use_wt:
                skip = self.iwt(skip)
                skip = self.upsample(skip)
                skip = self.dwt(skip)
            else:
                skip = self.upsample(skip)
            out = out + skip

        return out


class DualStyleUNet(nn.Module):
    def __init__(self, inp_size, inp_ch, out_ch, out_size, style_dim, n_mlp, middle_size=8, c_dim=0,
                 channel_multiplier=2, blur_kernel=[1, 3, 3, 1], lr_mlp=0.01):
        super().__init__()

        self.inp_size = inp_size

        self.style_dim = style_dim
        self.middle_log_size = int(math.log(middle_size, 2))

        layers = [PixelNorm()]
        if c_dim == 0:
            layers.append(EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
                ))
        else:
            layers.append(EqualLinear(
                    style_dim + c_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
                ))
        for i in range(n_mlp-1):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
                )
            )

        self.style = nn.Sequential(*layers)  # mapping network

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
            2048: 16 * channel_multiplier,
            4096: 16 * channel_multiplier
        }

        self.log_size = int(math.log(out_size, 2)) - 1

        # add new layer here
        # self.dwt = HaarTransform(3)
        # self.from_rgbs = nn.ModuleList()
        # self.cond_convs = nn.ModuleList()
        self.comb_convs = nn.ModuleList()

        in_channel = self.channels[inp_size // 2]  # 64
        self.from_rgbs = nn.ModuleList()
        self.cond_convs = nn.ModuleList()
        self.comb_convs = nn.ModuleList()   # 64, 32, 16
        self.comb_convs.append(ConvLayer(in_channel * 2, in_channel, 3))
        self.conv_in = ConvLayer(inp_ch, in_channel, 3, downsample=True)
        for i in range(int(math.log(inp_size, 2)) - 2, self.middle_log_size - 1, -1):  # 32, 16, 8
            out_channel = self.channels[2 ** i]  # (inp_size/2)->->(8*512)
            self.from_rgbs.append(FromRGB(in_channel, inp_ch, downsample=True, use_wt=False))  # //2
            # self.from_rgbs.append(FromRGB(in_channel, inp_ch, downsample=(i + 1)!=int(math.log(inp_size, 2)), use_wt=False))
            self.cond_convs.append(ConvBlock(in_channel, out_channel, blur_kernel))  # //2
            if i > self.middle_log_size:
                self.comb_convs.append(ConvLayer(out_channel * 2, out_channel, 3))
            else:
                self.comb_convs.append(ConvLayer(out_channel, out_channel, 3))  # 最后一层  (8*512)
            in_channel = out_channel

        # self.input = ConstantInput(self.channels[middle_size], size=middle_size)
        # self.conv1 = StyledConv(
        #     self.channels[middle_size], self.channels[middle_size], 3, style_dim, blur_kernel=blur_kernel
        # )
        # self.to_rgb1 = ToRGB(self.channels[middle_size], style_dim, upsample=False)

        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        self.to_rgbs1 = nn.ModuleList()
        self.to_rgbs2 = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[middle_size]

        self.num_layers = (self.log_size - self.middle_log_size) * 2
        for layer_idx in range(self.num_layers):
            res = (layer_idx + 8) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(*shape))

        for i in range(self.middle_log_size + 1, self.log_size + 1):  # 4, 5, 6, 7, 8, 9
            out_channel = self.channels[2 ** i]  # (16*512)->(32*512)->(64*512)->(128*256)->(256*128)->(512*64)

            self.convs1.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs1.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                )
            )

            self.to_rgbs1.append(ToRGB(in_channel=out_channel, style_dim=style_dim, out_channel=out_ch * 4))

            self.convs2.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample = True,
                    blur_kernel = blur_kernel,
                )
            )

            self.convs2.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel = blur_kernel
                )
            )

            self.to_rgbs2.append(ToRGB(in_channel = out_channel, style_dim = style_dim, out_channel = out_ch * 4))

            in_channel = out_channel
        self.iwt = InverseHaarTransform(out_ch)

        self.n_latent = self.log_size * 2 - (self.middle_log_size * 2 - 1) + 1

    def make_noise(self, device, zero_noise=False):
        noises = []
        func = torch.zeros if zero_noise else torch.randn
        for i in range(self.middle_log_size + 1, self.log_size + 1):
            for _ in range(2):
                noises.append(func(1, 1, 2 ** i, 2 ** i, device=device))
        # if zero_noise:
        #     for i in range(len(noises)):
        #         if i < len(noises) - 2:
        #             noises[i] = None
        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def forward(
            self,
            styles,
            condition_img,
            cond=None,
            return_latents=False,
            inject_index=None,
            truncation=1,
            truncation_latent=None,
            input_is_latent=False,
            noise=None,
            randomize_noise=True,
            view_feature1 = None,
            view_feature2 = None
    ):
        """
        :param randomize_noise: False, use fixed noise
        """
        if not input_is_latent:
            if cond is None:
                styles = [self.style(s) for s in styles]
            else:
                styles = [self.style(torch.cat([s, cond], dim=-1)) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)
                ]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        # cond_list = self.img_unet(condition_img)
        cond_img = condition_img
        cond_out = self.conv_in(cond_img)  ### None
        cond_list = [cond_out]  ### []
        cond_num = 0
        for from_rgb, cond_conv in zip(self.from_rgbs, self.cond_convs):
            cond_img, cond_out = from_rgb(cond_img, cond_out)
            cond_out = cond_conv(cond_out)
            # print('Down', cond_img.shape, cond_out.shape)
            cond_list.append(cond_out)
            cond_num += 1

        # out = self.input(latent)
        # out = self.conv1(out, latent[:, 0], noise=noise[0])
        # skip = self.to_rgb1(out, latent[:, 1])
        i = 0
        skip = None
        for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.convs1[::2], self.convs1[1::2], noise[::2], noise[1::2], self.to_rgbs1
        ):
            if i == 0:
                out = self.comb_convs[-1](cond_list[-1])
            elif i < 2 * len(self.comb_convs):
                out = torch.cat([out, cond_list[-1 - (i // 2)]], dim=1)
                out = self.comb_convs[-1 - (i // 2)](out)
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)
            # print(i, out.shape, skip.shape)
            if view_feature1 is not None and i == 8:
                view_feature1 = F.interpolate(view_feature1, out.shape[-2:], mode = 'bilinear')  # (B, 128, 256. 256)
                out = out + view_feature1
            i += 2
        image1 = self.iwt(skip)

        i = 0
        skip = None
        for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.convs2[::2], self.convs2[1::2], noise[::2], noise[1::2], self.to_rgbs2
        ):
            if i == 0:
                out = self.comb_convs[-1](cond_list[-1])
            elif i < 2 * len(self.comb_convs):
                out = torch.cat([out, cond_list[-1 - (i // 2)]], dim=1)
                out = self.comb_convs[-1 - (i // 2)](out)
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)
            if view_feature2 is not None and i == 8:
                view_feature2 = F.interpolate(view_feature2, out.shape[-2:], mode = 'bilinear')  # (B, 128, 256. 256)
                out = out + view_feature2
            i += 2
        image2 = self.iwt(skip)

        images = torch.cat([image1, image2], 1)

        if return_latents:
            return images, latent
        else:
            return images, None