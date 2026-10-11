"""Model 3: mini-Xception (Arriaga et al., 2017, "Real-time Convolutional Neural Networks
for Emotion and Gender Classification"), reimplemented in PyTorch."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2d(nn.Sequential):
    """Depthwise separable convolution: depthwise 3x3 then pointwise 1x1."""

    def __init__(self, c_in, c_out):
        super().__init__(
            nn.Conv2d(c_in, c_in, 3, padding=1, groups=c_in, bias=False),
            nn.Conv2d(c_in, c_out, 1, bias=False),
        )


class XceptionBlock(nn.Module):
    """Two separable convolutions + max-pooling, with a strided 1x1 residual shortcut."""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.residual = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride=2, bias=False), nn.BatchNorm2d(c_out))
        self.main = nn.Sequential(
            SeparableConv2d(c_in, c_out), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            SeparableConv2d(c_out, c_out), nn.BatchNorm2d(c_out),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

    def forward(self, x):
        return self.main(x) + self.residual(x)


class MiniXception(nn.Module):
    """width=1: original architecture (~57k parameters); width=2 doubles every channel count."""

    def __init__(self, num_classes=7, width=1):
        super().__init__()
        c = [8 * width, 16 * width, 32 * width, 64 * width, 128 * width]
        self.stem = nn.Sequential(
            nn.Conv2d(1, c[0], 3, bias=False), nn.BatchNorm2d(c[0]), nn.ReLU(inplace=True),
            nn.Conv2d(c[0], c[0], 3, bias=False), nn.BatchNorm2d(c[0]), nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[XceptionBlock(c[i], c[i + 1]) for i in range(4)])
        self.head = nn.Conv2d(c[4], num_classes, 3, padding=1)

    def forward(self, x):
        """x: (N, 1, 48, 48) in [0, 1]. Returns logits (N, num_classes)."""
        x = self.head(self.blocks(self.stem(x)))
        return x.mean(dim=(2, 3))  # global average pooling


@torch.no_grad()
def predict_proba(model, x, tta=True):
    """Softmax probabilities; with tta, averaged with the horizontally flipped image."""
    model.eval()
    p = torch.softmax(model(x), dim=1)
    if tta:
        p = (p + torch.softmax(model(x.flip(3)), dim=1)) / 2
    return p


def load_model(path, device="cpu"):
    """Load a {state_dict, width, tta} checkpoint saved by train_cnn."""
    ckpt = torch.load(path, map_location=device)
    model = MiniXception(width=ckpt["width"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt.get("tta", True)


def augment(x, max_rot=10, max_shift=0.1, max_zoom=0.1, max_bright=0.2):
    """Random per-batch augmentations, computed on GPU: horizontal flip, rotation,
    shift and zoom (a single affine transform), brightness/contrast."""
    n = x.shape[0]
    dev = x.device
    flip = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
    angle = (torch.rand(n, device=dev) * 2 - 1) * math.radians(max_rot)
    scale = 1 + (torch.rand(n, device=dev) * 2 - 1) * max_zoom
    shift = (torch.rand(n, 2, device=dev) * 2 - 1) * max_shift * 2  # normalized coordinates [-1, 1]

    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([
        torch.stack([cos * flip, -sin, shift[:, 0]], dim=1),
        torch.stack([sin * flip, cos, shift[:, 1]], dim=1),
    ], dim=1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="border", align_corners=False)

    bright = (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * max_bright
    contrast = 1 + (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * max_bright
    mean = x.mean(dim=(2, 3), keepdim=True)
    return ((x - mean) * contrast + mean + bright).clamp(0, 1)


def random_erasing(x, p=0.5, area=(0.02, 0.2), ratio=(0.3, 3.3)):
    """Erase (uniform noise) a random rectangle in a fraction p of the batch images."""
    n, _, h, w = x.shape
    dev = x.device
    s = torch.empty(n, device=dev).uniform_(*area) * h * w
    r = torch.exp(torch.empty(n, device=dev).uniform_(math.log(ratio[0]), math.log(ratio[1])))
    eh = (s * r).sqrt().clamp(1, h).long()
    ew = (s / r).sqrt().clamp(1, w).long()
    top = (torch.rand(n, device=dev) * (h - eh + 1)).long()
    left = (torch.rand(n, device=dev) * (w - ew + 1)).long()
    rows = torch.arange(h, device=dev).view(1, h, 1)
    cols = torch.arange(w, device=dev).view(1, 1, w)
    mask = ((rows >= top.view(-1, 1, 1)) & (rows < (top + eh).view(-1, 1, 1))
            & (cols >= left.view(-1, 1, 1)) & (cols < (left + ew).view(-1, 1, 1)))
    mask &= (torch.rand(n, device=dev) < p).view(-1, 1, 1)
    return torch.where(mask.unsqueeze(1), torch.rand_like(x), x)


def mix_batch(x, y, mixup_alpha=0.4, cutmix_alpha=1.0):
    """Apply mixup or cutmix (50/50): returns mixed x, y, permuted y and lambda."""
    perm = torch.randperm(x.shape[0], device=x.device)
    if torch.rand(1).item() < 0.5:
        lam = float(torch.distributions.Beta(mixup_alpha, mixup_alpha).sample())
        return lam * x + (1 - lam) * x[perm], y, y[perm], lam
    lam = float(torch.distributions.Beta(cutmix_alpha, cutmix_alpha).sample())
    h, w = x.shape[2:]
    ch, cw = int(h * math.sqrt(1 - lam)), int(w * math.sqrt(1 - lam))
    cy, cx = torch.randint(h, (1,)).item(), torch.randint(w, (1,)).item()
    y0, y1 = max(cy - ch // 2, 0), min(cy + ch // 2, h)
    x0, x1 = max(cx - cw // 2, 0), min(cx + cw // 2, w)
    x = x.clone()
    x[:, :, y0:y1, x0:x1] = x[perm, :, y0:y1, x0:x1]
    lam = 1 - (y1 - y0) * (x1 - x0) / (h * w)  # actual kept proportion
    return x, y, y[perm], lam
