"""Processed-RGB noise descriptors used by MicroDiffusion."""
import math

import torch
import torch.nn.functional as F


def _luminance(image):
    weights = image.new_tensor([0.2126, 0.7152, 0.0722])[None, :, None, None]
    return (image * weights).sum(1, keepdim=True)


def _nonnegative_line(means, variances):
    """Two-variable nonnegative least squares, separately for each RGB channel."""
    x = means[:, None]
    slope = ((x - x.mean(0)) * (variances - variances.mean(0))).sum(0)
    slope = slope / (x - x.mean(0)).square().sum(0).clamp_min(1e-12)
    intercept = variances.mean(0) - slope * x.mean(0)
    constant = variances.mean(0)
    through_origin = (x * variances).sum(0) / x.square().sum(0).clamp_min(1e-12)
    constant_error = (variances - constant).square().sum(0)
    origin_error = (variances - x * through_origin).square().sum(0)
    use_constant = constant_error <= origin_error
    valid = (slope >= 0) & (intercept >= 0)
    intercept = torch.where(valid, intercept, torch.where(use_constant, constant, torch.zeros_like(constant)))
    slope = torch.where(valid, slope, torch.where(use_constant, torch.zeros_like(slope), through_origin))
    return intercept, slope


@torch.no_grad()
def estimate_noise_statistics(low, cfg):
    """Return read_var[B,3], shot[B,3], correlation[B,3,3], support[B,1].

    The variance proxy is read_var + shot * processed-RGB luminance. HH uses
    coefficients (+1,-1,-1,+1)/2, hence unit noise energy for spatially independent
    noise. ISP correlations can change this relationship; it is not an unbiased
    estimator for arbitrary compressed images. Flat tiles are selected by their
    LOWPASS structure, never by small high-frequency residuals themselves.

    support is the fraction of complete tiles accepted for estimation, not a
    confidence probability. Ties at the flatness threshold are all retained so
    tile order cannot change the descriptors. Fewer than three luminance bins use a
    measured constant variance. No usable tiles (including too-small images)
    use std=4*min_std, bounded by [min_std,max_std], and zero shot coefficient.
    """
    block = int(cfg.block_size)
    if block < 8 or block % 2:
        raise ValueError("noise block_size must be even and >= 8")
    if not 0 < cfg.flat_fraction <= 1 or not 0 < cfg.min_std <= cfg.max_std:
        raise ValueError("Invalid flat_fraction or noise standard-deviation bounds")
    image = low.detach().float().clamp(0, 1)
    batch, _, height, width = image.shape
    minimum, maximum = cfg.min_std ** 2, cfg.max_std ** 2
    fallback = min(cfg.max_std, 4 * cfg.min_std) ** 2
    read = image.new_full((batch, 3), fallback)
    shot = image.new_zeros(batch, 3)
    identity = torch.eye(3, device=image.device, dtype=image.dtype)
    correlations = identity[None].repeat(batch, 1, 1)
    support = image.new_zeros(batch, 1)
    result = {"read_var": read, "shot": shot, "correlation": correlations, "support": support}
    if min(height, width) < block:
        return result

    tiles = image.unfold(2, block, block).unfold(3, block, block)
    tiles = tiles.permute(0, 2, 3, 1, 4, 5).reshape(batch, -1, 3, block, block)
    count = tiles.shape[1]
    flat = tiles.reshape(-1, 3, block, block)
    pooled = F.avg_pool2d(flat, 4, stride=4)
    gray = _luminance(pooled)
    structure = ((gray[:, :, 1:] - gray[:, :, :-1]).abs().mean((1, 2, 3)) +
                 (gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs().mean((1, 2, 3))) / 2
    structure = structure.reshape(batch, count)
    means = _luminance(flat).mean((1, 2, 3)).reshape(batch, count)
    clipped = ((flat <= 1 / 255) | (flat >= 254 / 255)).float().mean((1, 2, 3)).reshape(batch, count)
    eligible = (clipped < 0.05) & (means > 3 / 255) & (means < 0.97)
    hh = (tiles[..., 0::2, 0::2] - tiles[..., 0::2, 1::2] -
          tiles[..., 1::2, 0::2] + tiles[..., 1::2, 1::2]) / 2
    hh = hh.flatten(-2)
    centered = hh - hh.median(-1, keepdim=True).values
    deviations = centered.abs().median(-1).values / 0.6744897501960817
    variances = deviations.square()
    bins = torch.bucketize(means.contiguous(), means.new_tensor([0.08, 0.20, 0.45]))

    for index in range(batch):
        selected, bin_means, bin_variances = [], [], []
        for level in range(4):
            candidates = torch.nonzero(eligible[index] & (bins[index] == level), as_tuple=False).flatten()
            if candidates.numel() == 0:
                continue
            number = max(1, math.ceil(candidates.numel() * cfg.flat_fraction))
            scores = structure[index, candidates]
            threshold = scores.sort().values[number - 1]
            chosen = candidates[scores <= threshold]
            selected.append(chosen)
            bin_means.append(means[index, chosen].median())
            bin_variances.append(variances[index, chosen].median(0).values)
        if not selected:
            continue
        selected = torch.cat(selected)
        support[index, 0] = selected.numel() / count
        intensity = torch.stack(bin_means)
        variance = torch.stack(bin_variances)
        if len(bin_means) >= 3 and intensity.max() - intensity.min() >= 0.05:
            intercept, slope = _nonnegative_line(intensity, variance)
        else:
            intercept = variances[index, selected].median(0).values
            slope = torch.zeros_like(intercept)
        read[index] = intercept.clamp(minimum, maximum)
        shot[index] = slope.clamp_min(0).minimum((maximum - read[index]).clamp_min(0))

        # Per-tile standardization avoids assigning brighter/noisier tiles more
        # weight. Winsorization limits remaining edge outliers; 10% identity
        # shrinkage gives a strictly positive-definite RGB correlation matrix.
        standardized = centered[index, selected] / deviations[index, selected, :, None].clamp_min(cfg.min_std)
        samples = standardized.clamp(-4, 4).permute(1, 0, 2).reshape(3, -1)
        covariance = samples @ samples.t() / samples.shape[1]
        std = covariance.diag().clamp_min(1e-12).sqrt()
        correlation = covariance / (std[:, None] * std[None, :])
        correlation = (correlation + correlation.t()) / 2
        correlation = correlation * (1 - identity) + identity
        correlations[index] = 0.9 * correlation + 0.1 * identity
    return result


@torch.no_grad()
def noise_std_map(low, stats):
    """Approximate RGB std from a fixed 5x5 native-resolution luminance filter.

    Filtering is local and does not depend on crop/image size. The affine law
    describes processed-sRGB statistics; it is not a RAW Poisson-Gaussian claim.
    """
    gray = _luminance(low.detach().clamp(0, 1))
    gray = F.avg_pool2d(F.pad(gray, (2, 2, 2, 2), mode="replicate"), 5, stride=1)
    variance = stats["read_var"].to(low)[:, :, None, None] + stats["shot"].to(low)[:, :, None, None] * gray
    return variance.clamp_min(1e-12).sqrt()
