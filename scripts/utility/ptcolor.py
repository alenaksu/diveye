"""Pytorch routines for color conversions and management.

All color arguments are given as 4-dimensional tensors representing
batch of images (Bx3xHxW).  RGB values are supposed to be in the
range 0-1 (but values outside the range are tolerated).
"""

import torch
from torch import Tensor


def _t(data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.tensor(data, requires_grad=False, dtype=torch.float32, device=device)


def _mul(coeffs, image):
    coeffs = coeffs.to(image.device).view(3, 3, 1, 1)
    return torch.nn.functional.conv2d(image, coeffs)


_RGB_TO_XYZ = {
    "srgb": _t([[0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041]]),
    "prophoto": _t([[0.7976749, 0.1351917, 0.0313534],
                    [0.2880402, 0.7118741, 0.0000857],
                    [0.0000000, 0.0000000, 0.8252100]])
}

_XYZ_TO_RGB = {
    "srgb": _t([[3.2404542, -1.5371385, -0.4985314],
                [-0.9692660,  1.8760108,  0.0415560],
                [0.0556434, -0.2040259,  1.0572252]]),
    "prophoto": _t([[ 1.3459433, -0.2556075, -0.0511118],
                    [-0.5445989,  1.5081673,  0.0205351],
                    [0.0000000,  0.0000000,  1.2118128]])
}

WHITE_POINTS = {item[0]: _t(item[1:]).view(1, 3, 1, 1) for item in [
    ("a",   1.0985, 1.0000, 0.3558),
    ("b",   0.9807, 1.0000, 1.1822),
    ("e",   1.0000, 1.0000, 1.0000),
    ("d50", 0.9642, 1.0000, 0.8251),
    ("d55", 0.9568, 1.0000, 0.9214),
    ("d65", 0.9504, 1.0000, 1.0888),
    ("icc", 0.9642, 1.0000, 0.8249),
]}

_EPSILON = 0.008856
_KAPPA   = 903.3
_XYZ_TO_LAB = _t([[0.0, 116.0, 0.], [500.0, -500.0, 0.], [0.0, 200.0, -200.0]])
_LAB_TO_XYZ = _t([[1.0/116.0, 1.0/500.0, 0], [1.0/116.0, 0, 0], [1.0/116.0, 0, -1.0/200.0]])
_LAB_OFF = _t([16.0, 0.0, 0.0]).view(1, 3, 1, 1)


def apply_gamma(rgb, gamma="srgb"):
    if gamma == "srgb":
        T = 0.0031308
        rgb1 = torch.max(rgb, rgb.new_tensor(T))
        return torch.where(rgb < T, 12.92 * rgb, 1.055 * torch.pow(torch.abs(rgb1), 1/2.4) - 0.055)
    elif gamma is None:
        return rgb
    else:
        return torch.pow(torch.max(rgb, rgb.new_tensor(0.0)), 1.0 / gamma)


def remove_gamma(rgb, gamma="srgb"):
    if gamma == "srgb":
        T = 0.04045
        rgb1 = torch.max(rgb, rgb.new_tensor(T))
        return torch.where(rgb < T, rgb / 12.92, torch.pow(torch.abs(rgb1 + 0.055) / 1.055, 2.4))
    elif gamma is None:
        return rgb
    else:
        return (torch.pow(torch.max(rgb, rgb.new_tensor(0.0)), gamma)
                + torch.min(rgb, rgb.new_tensor(0.0)))


def rgb2xyz(rgb, gamma_correction="srgb", clip_rgb=False, space="srgb"):
    if clip_rgb:
        rgb = torch.clamp(rgb, 0, 1)
    rgb = remove_gamma(rgb, gamma_correction)
    return _mul(_RGB_TO_XYZ[space], rgb)


def xyz2rgb(xyz, gamma_correction="srgb", clip_rgb=False, space="srgb"):
    rgb = _mul(_XYZ_TO_RGB[space], xyz)
    if clip_rgb:
        rgb = torch.clamp(rgb, 0, 1)
    return apply_gamma(rgb, gamma_correction)


def _lab_f(x):
    x1 = torch.max(x, x.new_tensor(_EPSILON))
    return torch.where(x > _EPSILON, torch.pow(x1, 1.0/3), (_KAPPA * x + 16.0) / 116.0)


def xyz2lab(xyz, white_point="d65"):
    xyz = xyz / WHITE_POINTS[white_point].to(xyz.device)
    f_xyz = _lab_f(xyz)
    return _mul(_XYZ_TO_LAB, f_xyz) - _LAB_OFF.to(xyz.device)


def _inv_lab_f(x):
    x3 = torch.max(x, x.new_tensor(_EPSILON)) ** 3
    return torch.where(x3 > _EPSILON, x3, (116.0 * x - 16.0) / _KAPPA)


def lab2xyz(lab, white_point="d65"):
    f_xyz = _mul(_LAB_TO_XYZ, lab + _LAB_OFF.to(lab.device))
    xyz = _inv_lab_f(f_xyz)
    return xyz * WHITE_POINTS[white_point].to(lab.device)


def rgb2lab(rgb, white_point="d65", gamma_correction="srgb", clip_rgb=False, space="srgb"):
    return xyz2lab(rgb2xyz(rgb, gamma_correction, clip_rgb, space), white_point)


def lab2rgb(lab, white_point="d65", gamma_correction="srgb", clip_rgb=False, space="srgb"):
    return xyz2rgb(lab2xyz(lab, white_point), gamma_correction, clip_rgb, space)


def lab2lch(lab):
    l = lab[:, 0, :, :]
    c = torch.norm(lab[:, 1:, :, :], 2, 1)
    h = torch.atan2(lab[:, 2, :, :], lab[:, 1, :, :])
    h = h * (180 / 3.141592653589793)
    h = torch.where(h >= 0, h, 360 + h)
    return torch.stack([l, c, h], 1)


def rgb2lch(rgb, white_point="d65", gamma_correction="srgb", clip_rgb=False, space="srgb"):
    lab = rgb2lab(rgb, white_point, gamma_correction, clip_rgb, space)
    return lab2lch(lab)


def deltaE(lab1, lab2):
    return torch.norm(lab1 - lab2, 2, 1, keepdim=True)
