"""Modèle 3 : mini-Xception (Arriaga et al., 2017, « Real-time Convolutional Neural Networks
for Emotion and Gender Classification »), réimplémenté en PyTorch."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2d(nn.Sequential):
    """Convolution séparable en profondeur : depthwise 3x3 puis pointwise 1x1."""

    def __init__(self, c_in, c_out):
        super().__init__(
            nn.Conv2d(c_in, c_in, 3, padding=1, groups=c_in, bias=False),
            nn.Conv2d(c_in, c_out, 1, bias=False),
        )


class XceptionBlock(nn.Module):
    """Deux convolutions séparables + max-pooling, avec raccourci résiduel 1x1 de stride 2."""

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
    """width=1 : architecture d'origine (~57k paramètres) ; width=2 double tous les canaux."""

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
        """x : (N, 1, 48, 48) dans [0, 1]. Renvoie les logits (N, num_classes)."""
        x = self.head(self.blocks(self.stem(x)))
        return x.mean(dim=(2, 3))  # global average pooling


@torch.no_grad()
def predict_proba(model, x, tta=True):
    """Probabilités softmax ; avec tta, moyenne avec l'image retournée horizontalement."""
    model.eval()
    p = torch.softmax(model(x), dim=1)
    if tta:
        p = (p + torch.softmax(model(x.flip(3)), dim=1)) / 2
    return p


def load_model(path, device="cpu"):
    """Charge un checkpoint {state_dict, width, tta} sauvegardé par train_cnn."""
    ckpt = torch.load(path, map_location=device)
    model = MiniXception(width=ckpt["width"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt.get("tta", True)


def augment(x, max_rot=10, max_shift=0.1, max_zoom=0.1, max_bright=0.2):
    """Augmentations aléatoires par batch, directement sur GPU : flip horizontal,
    rotation, décalage, zoom (une seule transformation affine) et luminosité/contraste."""
    n = x.shape[0]
    dev = x.device
    flip = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
    angle = (torch.rand(n, device=dev) * 2 - 1) * math.radians(max_rot)
    scale = 1 + (torch.rand(n, device=dev) * 2 - 1) * max_zoom
    shift = (torch.rand(n, 2, device=dev) * 2 - 1) * max_shift * 2  # coordonnées normalisées [-1, 1]

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
