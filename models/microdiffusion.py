"""Weak-pair illumination transport and synthetic-pair noise removal.

The normal view provides scalar illumination statistics, never a real RGB
target. Noise removal is supervised with aligned synthetic low/low pairs.
The deterministic residual bridge is not a VP score model.
"""

import copy
from contextlib import contextmanager
import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import DiffusionUNet
from .components import (
    IlluminationPrior, MultiScaleRGBNet, box_filter, crop_together, gradients,
    highlight_safe_gain, resize, structure_confidence,
)
from .observation import estimate_noise_statistics, noise_std_map


def luminance(image):
    return (image * image.new_tensor([0.2126, 0.7152, 0.0722])[None, :, None, None]).sum(1, keepdim=True)


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {name: value.detach().clone() for name, value in self._model(model).named_parameters()
                       if value.requires_grad}

    @staticmethod
    def _model(model):
        return model.module if isinstance(model, nn.DataParallel) else model

    @torch.no_grad()
    def update(self, model):
        for name, value in self._model(model).named_parameters():
            if value.requires_grad:
                self.shadow[name].lerp_(value.detach(), 1 - self.decay)

    @torch.no_grad()
    def apply(self, model):
        for name, value in self._model(model).named_parameters():
            if value.requires_grad:
                value.copy_(self.shadow[name])

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state):
        self.shadow = {name: value.detach().clone() for name, value in state.items()}

    @contextmanager
    def average_parameters(self, model):
        module = self._model(model)
        backup = {name: value.detach().clone() for name, value in module.named_parameters() if value.requires_grad}
        self.apply(model)
        try:
            yield
        finally:
            with torch.no_grad():
                for name, value in module.named_parameters():
                    if name in backup:
                        value.copy_(backup[name])


ARCHITECTURE = "microdiffusion"
SINKHORN_ITERATIONS = 25
NOISE_ENDPOINT_PROBABILITY = 0.25
CORRECTION_FLOOR = 0.02
CORRECTION_SIGMAS = 3.0
EDGE_PROTECTION = 0.75


def image_mean(value, weight=None):
    """Keep the sample axis until after DataParallel gathers unequal shards."""
    if weight is None:
        return value.flatten(1).mean(1)
    weight = weight.expand_as(value)
    return (value * weight).flatten(1).sum(1) / weight.flatten(1).sum(1).clamp_min(1e-6)


@torch.no_grad()
def synthesize_low_pair(normal, low, observed_stats, cfg):
    """Known low-signal/noisy-low pair; real low contributes summaries only.

    Match the sampled real image's exposure approximately in processed RGB,
    with modest jitter and a small dark-tail augmentation. This is not a RAW
    sensor simulator. Smooth chroma and subsampling remain generic augmentations.
    """
    normal = normal.detach().clamp(0, 1)
    batch, _, height, width = normal.shape
    uniform = lambda: torch.rand(batch, 1, 1, 1, device=normal.device, dtype=normal.dtype)
    indices = torch.randint(low.shape[0], (batch,), device=low.device)
    level = luminance(low.detach()).mean((2, 3), keepdim=True)[indices]
    level = level * (0.85 + 0.30 * uniform())
    level = torch.where(uniform() < cfg.dark_tail_probability, level * (0.3 + 0.7 * uniform()), level)
    gain = level / luminance(normal).mean((2, 3), keepdim=True).clamp_min(0.02)
    field = torch.rand(batch, 1, 4, 4, device=normal.device, dtype=normal.dtype) - 0.5
    field = resize(field, (height, width))
    signal = (normal * gain.clamp(0.03, 1.2) * (0.3 * field).exp()).clamp(0, 1)
    identity = uniform() < cfg.identity_probability
    signal = torch.where(identity, normal, signal)
    variation = (0.5 + uniform()).flatten(1)
    stats = {key: observed_stats[key][indices] * variation for key in ("read_var", "shot")}
    correlation = observed_stats["correlation"][indices]
    epsilon = torch.einsum("bij,bjhw->bihw", torch.linalg.cholesky(correlation), torch.randn_like(signal))
    noisy = signal + noise_std_map(signal, stats) * epsilon
    chroma = torch.randn(batch, 3, max(2, math.ceil(height / 16)), max(2, math.ceil(width / 16)),
                         device=normal.device, dtype=normal.dtype)
    chroma = resize(chroma, (height, width))
    noisy = noisy + cfg.chroma_std * uniform() * (chroma - luminance(chroma))
    gray = luminance(noisy)
    subsampled = gray + resize(F.avg_pool2d(noisy - gray, 2, stride=2, ceil_mode=True), (height, width))
    noisy = torch.where(uniform() < 0.5, subsampled, noisy)
    noisy = (noisy.clamp(0, 1) * 255).round() / 255
    return signal, torch.where(identity, normal, noisy)


class ReliableSceneTeacher(nn.Module):
    """Transport scalar light; use a low-only fallback if the reference fails."""

    def __init__(self, cfg):
        nn.Module.__init__(self)
        self.grid = cfg.grid_size
        self.iterations = SINKHORN_ITERATIONS
        self.entropy = cfg.entropy
        self.max_log_gain = math.log(cfg.max_gain)
        self.fallback_mean = cfg.fallback_mean
        self.variant = 'full'

    @staticmethod
    def _ranks(values):
        return values.argsort(1).argsort(1).to(values.dtype) / max(values.shape[1] - 1, 1)

    def _patches(self, image):
        # Native-resolution filtering precedes the scene grid. Do not treat the
        # normal view at the same pixel coordinates as a restoration target.
        gray = luminance(image)
        light = box_filter(gray, 15).clamp_min(0.02)
        reflectance = box_filter(gray, 3) / light
        gx, gy = gradients(reflectance)
        pool = lambda x: F.adaptive_avg_pool2d(x, (self.grid, self.grid)).flatten(2).transpose(1, 2)
        descriptors = torch.cat((pool(reflectance), pool(gx.abs()), pool(gy.abs())), dim=2)
        quality = pool(((gray > 0.02) & (image.max(1, keepdim=True).values < 0.95)).float()).squeeze(-1)
        border = torch.ones_like(quality).reshape(-1, self.grid, self.grid)
        border[:, (0, -1)] = 0  # Timestamp overlays provide no scene evidence.
        return pool(light).squeeze(-1), descriptors, quality * border.flatten(1)

    @staticmethod
    def _quantile_target(low_l, low_q, high_l, high_q):
        low_order, high_order = low_l.argsort(1), high_l.argsort(1)
        low_weights = low_q.gather(1, low_order).clamp_min(1e-4)
        ranks = (low_weights.cumsum(1) - 0.5 * low_weights) / low_weights.sum(1, keepdim=True)
        ranks = ranks.gather(1, low_order.argsort(1))
        high_weights = high_q.gather(1, high_order)
        cdf = high_weights.cumsum(1) / high_weights.sum(1, keepdim=True).clamp_min(1e-6)
        indices = torch.searchsorted(cdf.contiguous(), ranks.contiguous()).clamp_max(high_l.shape[1] - 1)
        return high_l.gather(1, high_order).gather(1, indices)

    @torch.no_grad()
    def forward(self, low, normal, size):
        low_l, low_d, low_q = self._patches(low)
        if self.variant == 'low_only_prior':
            gain = (self.fallback_mean / low_l.clamp_min(0.02)).log().clamp(0, self.max_log_gain)
            gain = gain.reshape(-1, 1, self.grid, self.grid)
            gain = torch.cat((gain[:, :, 1:2], gain[:, :, 1:-1], gain[:, :, -2:-1]), dim=2)
            return resize(gain, size), low.new_zeros(low.shape[0], 1, *size)
        high_l, high_d, high_q = self._patches(normal)
        coordinates = torch.linspace(0, 1, self.grid, device=low.device)
        yy = coordinates[:, None].expand(-1, self.grid)
        xx = coordinates[None, :].expand(self.grid, -1)
        points = torch.stack((yy.flatten(), xx.flatten()), dim=1)
        spatial = (points[:, None] - points[None, :]).square().sum(-1)
        distance = (low_d[:, :, None] - high_d[:, None, :]).abs().mean(-1)
        if self.variant == 'rank_only_ot':
            spatial = torch.zeros_like(spatial)
            distance = torch.zeros_like(distance)
        reliability = torch.exp(-distance / 0.15) * torch.exp(-spatial[None] / 0.3)
        cost = 0.3 * (self._ranks(low_l)[:, :, None] - self._ranks(high_l)[:, None, :]).abs()
        cost = cost + spatial[None] + distance
        a = (low_q * reliability.max(2).values).clamp_min(1e-4)
        b = (high_q * reliability.max(1).values).clamp_min(1e-4)
        log_a = (a / a.sum(1, keepdim=True)).log()
        log_b = (b / b.sum(1, keepdim=True)).log()
        kernel = -cost / self.entropy
        u, v = torch.zeros_like(a), torch.zeros_like(b)
        for _ in range(self.iterations):
            u = log_a - torch.logsumexp(kernel + v[:, None], dim=2)
            v = log_b - torch.logsumexp(kernel + u[:, :, None], dim=1)
        coupling = torch.softmax(kernel + u[:, :, None] + v[:, None], dim=2)
        matched = (coupling * reliability * high_q[:, None]).sum(2)
        concentration = 1 + (coupling * coupling.clamp_min(1e-8).log()).sum(2) / math.log(self.grid ** 2)
        confidence = (low_q * matched * concentration.clamp(0, 1)).clamp(0, 1)
        scene_confidence = (low_q * matched).sum(1, keepdim=True) / low_q.sum(1, keepdim=True).clamp_min(1e-6)
        transported = (coupling * high_l[:, None]).sum(2)
        quantile = self._quantile_target(low_l, low_q, high_l, high_q)
        to_gain = lambda target: (target.clamp_min(0.02) / low_l.clamp_min(0.02)).log().clamp(0, self.max_log_gain)
        fallback = to_gain(torch.full_like(low_l, self.fallback_mean))
        global_gain = scene_confidence * to_gain(quantile) + (1 - scene_confidence) * fallback
        gain = confidence * to_gain(transported) + (1 - confidence) * global_gain
        if self.variant == 'no_reliability':
            gain = to_gain(transported)
            confidence = torch.ones_like(confidence)
        gain = gain.reshape(-1, 1, self.grid, self.grid)
        # Excluded overlay rows have no light target of their own. Extend the
        # interior gain instead of lifting them independently to fallback_mean,
        # which produced a bright bottom band in the native-image short run.
        gain = torch.cat((gain[:, :, 1:2], gain[:, :, 1:-1], gain[:, :, -2:-1]), dim=2)
        return (resize(gain, size),
                resize(confidence.reshape(-1, 1, self.grid, self.grid), size))


class MicroDiffusionNet(nn.Module):
    """Reuse the small scalar brightness path, replacing the RGB training task."""

    def __init__(self, args, config):
        nn.Module.__init__(self)
        self.config = config
        self.max_log_gain = math.log(config.prior.max_gain)
        self.noise = MultiScaleRGBNet(9, config.model.noise_width, timed=True)
        # Zero feature padding produced visible border bands in the real-image
        # smoke run. Keep constant regions constant at each pyramid boundary.
        for layer in self.noise.modules():
            if isinstance(layer, nn.Conv2d) and layer.padding != (0, 0):
                layer.padding_mode = "replicate"
        nn.init.zeros_(self.noise.output.weight)
        nn.init.zeros_(self.noise.output.bias)
        self.prior = IlluminationPrior(config.model.ch, self.max_log_gain)
        self.teacher = ReliableSceneTeacher(config.prior)
        self.teacher.variant = config.ablation.variant
        light_config = copy.deepcopy(config)
        light_config.data.conditional = False
        light_config.model.in_channels, light_config.model.out_ch = 8, 1
        self.brightness = DiffusionUNet(light_config)
        if config.ablation.variant == 'no_refinement':
            self.brightness.requires_grad_(False)
        nn.init.zeros_(self.brightness.conv_out.weight)
        nn.init.zeros_(self.brightness.conv_out.bias)

    def _working_size(self, image):
        height, width = image.shape[-2:]
        limit = self.config.model.illumination_size
        factor = min(1.0, limit / max(height, width))
        divisor = 2 ** (len(self.config.model.ch_mult) - 1)
        return tuple(max(divisor, int(math.ceil(size * factor / divisor)) * divisor)
                     for size in (height, width))

    @torch.no_grad()
    def _observations(self, low, stats=None):
        if stats is None:
            stats = estimate_noise_statistics(low, self.config.noise)
        noise = noise_std_map(low, stats)
        light = box_filter(luminance(low), 15).clamp_min(0.02)
        reflectance = low / light
        structure = structure_confidence(reflectance, noise.mean(1, keepdim=True), light)
        return light, reflectance, noise, structure

    def _correction_bound(self, noise, structure):
        if self.config.ablation.variant == 'fixed_noise_bound':
            noise = torch.full_like(noise, self.config.ablation.fixed_noise_std)
        if self.config.ablation.variant == 'no_edge_bound':
            structure = torch.zeros_like(structure)
        bound = (CORRECTION_FLOOR + CORRECTION_SIGMAS * noise).clamp(
            max=self.config.noise.max_correction)
        return bound * (1 - EDGE_PROTECTION * structure)

    def _residual(self, state, time, low_log_light, reflectance, gain, confidence, structure):
        if self.config.ablation.variant == 'no_refinement':
            return -gain
        condition = torch.cat((state, low_log_light, reflectance, gain, confidence, structure), dim=1)
        correction = 0.25 * self.max_log_gain * torch.tanh(self.brightness(condition, 1000 * time))
        return -(gain + correction).clamp(0, self.max_log_gain)

    def _predict_clean(self, state, time, observation, noise, structure):
        inputs = torch.cat((state, observation, noise / self.config.noise.max_std), dim=1)
        correction = self._correction_bound(noise, structure) * torch.tanh(self.noise(inputs, time))
        # The bound is relative to the fixed observation at EVERY step, so the
        # final low-domain correction cannot accumulate with the step count.
        return (observation + correction).clamp(0, 1)

    @torch.no_grad()
    def _denoise(self, low, noise, structure):
        state = low.clone()
        steps = self.config.diffusion.noise_steps
        for index in range(steps, 0, -1):
            time = low.new_full((low.shape[0],), index / steps)
            clean = self._predict_clean(state, time, low, noise, structure)
            ratio = (index - 1) / index
            state = ratio * state + (1 - ratio) * clean
        return state

    def _noise_losses(self, low, normal, stats):
        signal, noisy = synthesize_low_pair(normal, low, stats, self.config.degradation)
        _, _, noise, structure = self._observations(noisy)
        synthetic_luma = image_mean(luminance(noisy))
        signal, noisy, noise, structure = crop_together((signal, noisy, noise, structure), self.config.data.patch_size)
        batch = low.shape[0]
        time = torch.rand(batch, device=low.device) * 0.95 + 0.05
        time = torch.where(torch.rand_like(time) < NOISE_ENDPOINT_PROBABILITY,
                           torch.ones_like(time), time)
        state = (1 - time[:, None, None, None]) * signal + time[:, None, None, None] * noisy
        clean = self._predict_clean(state, time, noisy, noise, structure)
        error = ((clean - signal).square() + 1e-6).sqrt() / noise.clamp_min(0.01)
        synthetic_loss = image_mean(error, 1 + 2 * structure)

        return {"synthetic": synthetic_loss,
                "synthetic_endpoint_l1": image_mean((clean.detach() - signal).abs()),
                "synthetic_luma": synthetic_luma,
                "synthetic_correction_rms": image_mean((clean.detach() - noisy).square()).sqrt()}

    def forward(self, inputs):
        if inputs.ndim != 4 or inputs.shape[1] not in (3, 6) or min(inputs.shape[-2:]) < 16:
            raise ValueError("MicroDiffusion expects Bx3xHxW or Bx6xHxW images with H,W >= 16")
        if self.training and inputs.shape[1] != 6:
            raise ValueError("Training requires a same-scene low/normal weak pair (6 channels)")
        low = inputs[:, :3].detach()
        stats = estimate_noise_statistics(low, self.config.noise)
        light, reflectance, noise, structure = self._observations(low, stats)
        size = self._working_size(low)
        gain, confidence = self.prior(*[resize(value, size) for value in (low, light, noise.mean(1, keepdim=True), structure)])
        low_log, small_r, small_s = resize(light, size).log(), resize(reflectance, size), resize(structure, size)
        if not self.training:
            denoised = self._denoise(low, noise, structure)
            log_state = low_log.clone()
            for index in range(self.config.diffusion.brightness_steps, 0, -1):
                time = low.new_full((low.shape[0],), index / self.config.diffusion.brightness_steps)
                log_state = log_state - self._residual(log_state, time, low_log, small_r, gain, confidence, small_s) / self.config.diffusion.brightness_steps
            raw_gain = resize((log_state - low_log).clamp(0, self.max_log_gain), low.shape[-2:])
            output_gain = highlight_safe_gain(raw_gain, denoised)
            return {"pred_x": (denoised * output_gain.exp()).clamp(0, 1), "low_signal": denoised,
                    "enhancement_gain": output_gain / self.max_log_gain,
                    "noise_std": noise / self.config.noise.max_std,
                    "prior_confidence": resize(confidence, low.shape[-2:]), "structure_confidence": structure}

        normal = inputs[:, 3:6].detach()
        target_gain, target_confidence = self.teacher(low, normal, size)
        reliability = 0.2 + 0.8 * target_confidence
        if self.config.ablation.variant == 'no_reliability':
            reliability = torch.ones_like(target_confidence)
        prior_loss = image_mean((gain - target_gain).abs(), reliability)
        prior_loss = prior_loss + 0.1 * image_mean(F.binary_cross_entropy(confidence, target_confidence, reduction="none"))
        time = torch.rand(low.shape[0], device=low.device)
        time = torch.where(torch.rand_like(time) < 0.25, torch.ones_like(time), time)
        state = low_log + (1 - time[:, None, None, None]) * target_gain
        residual = self._residual(state, time, low_log, small_r, gain.detach(), confidence.detach(), small_s)
        transport_loss = image_mean((residual + target_gain).abs(), reliability)
        terms = self._noise_losses(low, normal, stats)
        terms.update({"prior": prior_loss, "transport": transport_loss,
                      "mean_gain": image_mean(gain.detach().exp()), "mean_confidence": image_mean(target_confidence),
                      "observed_luma": image_mean(luminance(low)), "observed_noise_std": image_mean(noise),
                      "observed_support": stats["support"].flatten()})
        return terms


class MicroDiffusion:
    """Model, optimizer, EMA, and checkpoint handling."""

    def __init__(self, args, config):
        self.args, self.config, self.device = args, config, config.device
        self._validate_config()
        model = MicroDiffusionNet(args, config).to(self.device)
        self.model = nn.DataParallel(model) if self.device.type == "cuda" else model
        self.ema_helper = EMA(self.model, config.model.ema_rate)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.optim.lr, weight_decay=config.optim.weight_decay,
            betas=(0.9, 0.999), amsgrad=config.optim.amsgrad, eps=config.optim.eps)
        self.start_epoch, self.step, self.last_loss_terms = 0, 0, {}
        print("MicroDiffusion: {:.3f}M parameters".format(sum(p.numel() for p in model.parameters()) / 1e6))

    def _validate_config(self):
        cfg = self.config
        if cfg.model.architecture != ARCHITECTURE:
            raise ValueError("Use the microdiffusion architecture")
        if cfg.model.ch < 32 or cfg.model.ch % 32 or cfg.model.noise_width < 8 or cfg.model.noise_width % 8:
            raise ValueError("Brightness width must be a multiple of 32; noise_width a multiple of 8")
        if cfg.data.patch_size < 32 or cfg.data.patch_size % 8 or cfg.model.illumination_size < 16:
            raise ValueError("patch_size must be a multiple of 8 >= 32; illumination_size >= 16")
        if cfg.prior.grid_size < 3 or cfg.prior.entropy <= 0 or cfg.prior.max_gain <= 1 or not 0 < cfg.prior.fallback_mean < 1:
            raise ValueError("Invalid scalar scene-prior configuration")
        if min(cfg.diffusion.noise_steps, cfg.diffusion.brightness_steps) < 1:
            raise ValueError("Sampling step counts must be positive")
        n = cfg.noise
        if n.block_size < 8 or n.block_size % 2 or not 0 < n.flat_fraction <= 1 or not 0 < n.min_std <= n.max_std:
            raise ValueError("Invalid native-resolution noise statistics")
        if not CORRECTION_FLOOR <= n.max_correction <= 1:
            raise ValueError("max_correction must cover the correction floor and not exceed one")
        d = cfg.degradation
        if d.chroma_std < 0 or not 0 <= d.identity_probability < 1 or not 0 <= d.dark_tail_probability < 1:
            raise ValueError("Invalid synthetic low-domain augmentation")
        if cfg.training.warmup_steps < 0 or cfg.training.ramp_steps < 1 or not 0 < cfg.model.ema_rate < 1:
            raise ValueError("Invalid warmup/ramp/EMA settings")
        if set(vars(cfg.loss)) != {"synthetic", "prior", "transport"} or any(v < 0 for v in vars(cfg.loss).values()):
            raise ValueError("Use nonnegative synthetic/prior/transport loss weights")

    def noise_estimation_loss(self, output):
        terms = {name: value.mean() for name, value in output.items()}
        cfg = self.config
        if cfg.ablation.variant == 'no_eta_ramp':
            ramp = 1.0
        else:
            ramp = min(1.0, max(0.0, (self.step - cfg.training.warmup_steps) / cfg.training.ramp_steps))
        noise = cfg.loss.synthetic * terms["synthetic"]
        guidance = cfg.loss.prior * terms["prior"] + ramp * cfg.loss.transport * terms["transport"]
        self.last_loss_terms = {name: value.detach() for name, value in terms.items()}
        self.last_loss_terms["ramp"] = noise.new_tensor(ramp)
        return noise, guidance

    def initialize_from_v4(self, path):
        options = {"map_location": self.device}
        if "weights_only" in inspect.signature(torch.load).parameters:
            options["weights_only"] = False
        checkpoint = torch.load(path, **options)
        saved = checkpoint["config"]
        if (getattr(saved.model, "diffusion_class", None) != "MicroDiffusionV4" or
                getattr(saved.model, "architecture", None) != "scene_prior_guarded_bridge_v4"):
            raise ValueError("Warm start requires a genuine v4 checkpoint")
        fields = (("model", ("ch", "noise_width", "ch_mult", "num_res_blocks", "illumination_size")),
                  ("prior", tuple(vars(self.config.prior))), ("noise", tuple(vars(self.config.noise))))
        for section, names in fields:
            for name in names:
                if getattr(getattr(saved, section), name, None) != getattr(getattr(self.config, section), name):
                    raise ValueError("Checkpoint process mismatch: {}.{}".format(section, name))
        state = {(key[7:] if key.startswith("module.") else key): value
                 for key, value in checkpoint["state_dict"].items()}
        ema = {(key[7:] if key.startswith("module.") else key): value
               for key, value in checkpoint.get("ema_helper", {}).items()}
        state.update(ema)
        inner = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        incompatible = inner.load_state_dict(state, strict=False)
        if incompatible.unexpected_keys:
            raise ValueError("Unexpected v4/v7 warm-start parameters: {}".format(incompatible.unexpected_keys[:3]))
        self.ema_helper = EMA(self.model, self.config.model.ema_rate)
        print("Initialized compatible parameters from the v4 checkpoint")

    def load_ddm_ckpt(self, load_path, ema=False, load_optimizer=False):
        options = {"map_location": self.device}
        if "weights_only" in inspect.signature(torch.load).parameters:
            options["weights_only"] = False
        checkpoint = torch.load(load_path, **options)
        saved = checkpoint.get("config")
        if (getattr(getattr(saved, "model", None), "diffusion_class", None) != "MicroDiffusion" or
                getattr(saved.model, "architecture", None) != ARCHITECTURE):
            raise ValueError("Checkpoint is not a compatible MicroDiffusion checkpoint")
        fields = (("model", ("ch", "noise_width", "ch_mult", "num_res_blocks", "illumination_size")),
                  ("prior", tuple(vars(self.config.prior))), ("noise", tuple(vars(self.config.noise))))
        for section, names in fields:
            for name in names:
                if getattr(getattr(saved, section), name, None) != getattr(getattr(self.config, section), name):
                    raise ValueError("Checkpoint process mismatch: {}.{}".format(section, name))
        state = {(k[7:] if k.startswith("module.") else k): v for k, v in checkpoint["state_dict"].items()}
        inner = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        inner.load_state_dict(state, strict=True)
        self.step, self.start_epoch = checkpoint.get("step", 0), checkpoint.get("epoch", 0)
        if load_optimizer and not checkpoint.get("epoch_complete", False):
            self.start_epoch = max(0, self.start_epoch - 1)
        if checkpoint.get("ema_helper"):
            self.ema_helper.load_state_dict(checkpoint["ema_helper"])
        else:
            self.ema_helper = EMA(self.model, self.config.model.ema_rate)
        if load_optimizer and checkpoint.get("optimizer") is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if ema:
            self.ema_helper.apply(self.model)
        print("Loaded MicroDiffusion checkpoint {} (step {}, epoch {}, EMA={})".format(
            load_path, self.step, self.start_epoch, ema))
        return checkpoint
