"""Core illumination and RGB modules used by MicroDiffusion."""
import math

import torch
from torch import nn
import torch.nn.functional as F

from .backbone import get_timestep_embedding


def luminance(image):
    return 0.299 * image[:, :1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]


def box_filter(image, size):
    pad = size // 2
    return F.avg_pool2d(F.pad(image, (pad, pad, pad, pad), mode="replicate"), size, 1)


def resize(image, size):
    return F.interpolate(image, size=size, mode="bilinear", align_corners=False)


def gradients(image):
    kernel = image.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    kernels = torch.stack((kernel, kernel.t()))[:, None]
    xy = F.conv2d(F.pad(image, (1, 1, 1, 1), mode="replicate"), kernels)
    return xy[:, :1], xy[:, 1:2]


@torch.no_grad()
def structure_confidence(reflectance, sigma, light):
    gx, gy = gradients(luminance(reflectance))
    xx, yy, xy = box_filter(gx.square(), 5), box_filter(gy.square(), 5), box_filter(gx * gy, 5)
    energy = xx + yy
    coherence = ((xx - yy).square() + 4 * xy.square() + 1e-12).sqrt() / (energy + 1e-6)
    strength = energy.sqrt() / (energy.sqrt() + 2 * sigma / light + 1e-6)
    return (coherence * strength).clamp(0, 1)


class IlluminationPrior(nn.Module):
    """Distil the training-only normal-light prior into a low-only predictor."""

    def __init__(self, width, max_log_gain):
        super().__init__()
        self.max_log_gain = max_log_gain
        self.local = nn.Sequential(nn.Conv2d(6, width, 3, padding=1), nn.SiLU(),
                                   nn.Conv2d(width, width, 3, padding=1), nn.SiLU())
        self.scene = nn.Conv2d(width, width, 1)
        self.output = nn.Conv2d(width, 2, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.constant_(self.output.bias, -1.5)

    def forward(self, image, light, sigma, structure):
        feature = self.local(torch.cat((image, light, sigma / 0.06, structure), dim=1))
        feature = feature + self.scene(F.adaptive_avg_pool2d(feature, 1))
        prediction = torch.sigmoid(self.output(feature))
        return self.max_log_gain * prediction[:, :1], prediction[:, 1:2]


def protected_gain(log_gain, image):
    """Smoothly reduce gain at existing light sources and saturated sky."""
    brightness = box_filter(image.max(dim=1, keepdim=True).values, 5)
    return log_gain * (1 - torch.sigmoid((brightness - 0.75) / 0.08))


def highlight_safe_gain(log_gain, signal):
    """Keep shadow gain and roll off highlights with one shared RGB multiplier.

    The original mask only protects already-bright inputs. A moderate input
    can still exceed one after amplification, causing channel-wise clipping.
    Start a smooth shoulder at 0.9 (or the existing peak, if brighter).
    """
    requested = protected_gain(log_gain, signal)
    peak = signal.max(1, keepdim=True).values.clamp(1e-6, 1)
    proposed_peak = peak * requested.exp()
    knee = peak.clamp_min(0.9)
    headroom = (1 - knee).clamp_min(1e-6)
    shoulder = knee - headroom * torch.expm1(-(proposed_peak - knee).clamp_min(0) / headroom)
    bounded_peak = torch.where(proposed_peak > knee, shoulder, proposed_peak).clamp_max(1)
    return (bounded_peak / peak).log().clamp_min(0)


def crop_together(values, patch_size):
    """Keep synthetic pairs/maps aligned, including the stride-8 pyramid phase."""
    batch, _, height, width = values[0].shape
    height, width = min(height, patch_size), min(width, patch_size)
    crops = [[] for _ in values]
    for index in range(batch):
        top = 8 * torch.randint((values[0].shape[-2] - height) // 8 + 1, ()).item()
        left = 8 * torch.randint((values[0].shape[-1] - width) // 8 + 1, ()).item()
        for result, value in zip(crops, values):
            result.append(value[index:index + 1, :, top:top + height, left:left + width])
    return [torch.cat(result, dim=0) for result in crops]


class ChannelNorm(nn.Module):
    """Normalize feature channels at each pixel, not spatial image statistics."""

    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, value):
        centered = value - value.mean(1, keepdim=True)
        return centered * torch.rsqrt(centered.square().mean(1, keepdim=True) + 1e-6) * self.weight + self.bias


class RGBBlock(nn.Module):
    def __init__(self, channels, time_width=0):
        super().__init__()
        self.norm1, self.norm2 = ChannelNorm(channels), ChannelNorm(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.affine = nn.Linear(time_width, 2 * channels) if time_width else None

    def forward(self, value, embedding):
        hidden = self.norm2(self.conv1(F.silu(self.norm1(value))))
        if self.affine is not None:
            scale, shift = self.affine(F.silu(embedding)).chunk(2, dim=1)
            hidden = hidden * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return value + 0.2 * self.conv2(F.silu(hidden))


class MultiScaleRGBNet(nn.Module):
    """Lossless 2x packing followed by a three-scale RGB U-Net.

    Packing preserves the input samples; learned convolutions can still change
    thin structures. Crop origins must share the full stride-8 lattice.
    """

    def __init__(self, in_channels, width, timed=False):
        super().__init__()
        self.width = width
        time_width = 4 * width if timed else 0
        self.time = (nn.Sequential(nn.Linear(width, time_width), nn.SiLU(),
                                   nn.Linear(time_width, time_width)) if timed else None)

        def blocks(channels):
            return nn.ModuleList([RGBBlock(channels, time_width) for _ in range(2)])

        self.input = nn.Conv2d(4 * in_channels, width, 3, padding=1)
        self.enc0, self.enc1, self.middle = blocks(width), blocks(2 * width), blocks(4 * width)
        self.down0 = nn.Conv2d(width, 2 * width, 3, stride=2, padding=1)
        self.down1 = nn.Conv2d(2 * width, 4 * width, 3, stride=2, padding=1)
        self.up1 = nn.Conv2d(4 * width, 2 * width, 3, padding=1)
        self.merge1 = nn.Conv2d(4 * width, 2 * width, 1)
        self.dec1 = blocks(2 * width)
        self.up0 = nn.Conv2d(2 * width, width, 3, padding=1)
        self.merge0 = nn.Conv2d(2 * width, width, 1)
        self.dec0 = blocks(width)
        self.output = nn.Conv2d(width, 12, 3, padding=1)
        nn.init.normal_(self.output.weight, std=0.001)
        nn.init.zeros_(self.output.bias)

    def forward(self, inputs, time=None):
        height, width = inputs.shape[-2:]
        inputs = F.pad(inputs, (0, (-width) % 8, 0, (-height) % 8), mode="replicate")
        embedding = self.time(get_timestep_embedding(1000 * time, self.width)) if self.time is not None else None

        def run(blocks, value):
            for block in blocks:
                value = block(value, embedding)
            return value

        skip0 = run(self.enc0, self.input(F.pixel_unshuffle(inputs, 2)))
        skip1 = run(self.enc1, self.down0(skip0))
        feature = run(self.middle, self.down1(skip1))
        feature = self.up1(F.interpolate(feature, size=skip1.shape[-2:], mode="nearest"))
        feature = run(self.dec1, self.merge1(torch.cat((feature, skip1), dim=1)))
        feature = self.up0(F.interpolate(feature, size=skip0.shape[-2:], mode="nearest"))
        feature = run(self.dec0, self.merge0(torch.cat((feature, skip0), dim=1)))
        return F.pixel_shuffle(self.output(F.silu(feature)), 2)[:, :, :height, :width]
