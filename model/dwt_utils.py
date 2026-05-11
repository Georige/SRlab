"""Haar Discrete Wavelet Transform using pixel_unshuffle + matrix multiply.

No external dependency — pure PyTorch, GPU-efficient, gradient-friendly.

Formulation:
  Given 2×2 patches [a00, a01, a10, a11] per channel:
    LL = (a00 + a01 + a10 + a11) / 2   (low-frequency approximation)
    LH = (a00 - a01 + a10 - a11) / 2   (horizontal details)
    HL = (a00 + a01 - a10 - a11) / 2   (vertical details)
    HH = (a00 - a01 - a10 + a11) / 2   (diagonal details)

  Forward and inverse use the same symmetric Haar matrix H/2 where H@H = 4I.
"""

import torch
import torch.nn.functional as F


# Haar 4×4 transform matrix. Symmetric, H@H = 4I.
# Rows: [LL, LH, HL, HH], Cols: [a00, a01, a10, a11]
def _haar_matrix(dtype, device):
    return torch.tensor(
        [[1.,  1.,  1.,  1.],
         [1., -1.,  1., -1.],
         [1.,  1., -1., -1.],
         [1., -1., -1.,  1.]],
        dtype=dtype, device=device) / 2.0


def dwt_haar(x):
    """Haar 2D DWT.

    Args:
        x: [B, C, H, W]. H and W must be even.

    Returns:
        LL, LH, HL, HH: each [B, C, H//2, W//2].
    """
    B, C, H, W = x.shape
    # pixel_unshuffle: group 2×2 patches → 4*C channels
    # Output layout: [a00, a01, a10, a11] per spatial position
    patches = F.pixel_unshuffle(x, 2)           # [B, C*4, H//2, W//2]
    patches = patches.view(B, C, 4, H // 2, W // 2)  # [B, C, 4, H', W']

    H_mat = _haar_matrix(x.dtype, x.device)     # [4, 4]
    # y[b,c,o,h,w] = sum_p patches[b,c,p,h,w] * H[o,p]
    y = torch.einsum('bcphw,op->bcohw', patches, H_mat)  # [B, C, 4, H', W']

    return y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3]


def idwt_haar(LL, LH, HL, HH):
    """Inverse Haar 2D DWT.

    Args:
        LL, LH, HL, HH: each [B, C, H', W'].

    Returns:
        Reconstruction: [B, C, H'*2, W'*2].
    """
    B, C, H, W = LL.shape
    y = torch.stack([LL, LH, HL, HH], dim=2)    # [B, C, 4, H, W]

    H_mat = _haar_matrix(y.dtype, y.device)     # [4, 4]
    # Inverse uses same H/2 (since H@H=4I, so H/2 @ H/2 @ x = x/4 * 4 = x)
    patches = torch.einsum('bcohw,po->bcphw', y, H_mat)  # [B, C, 4, H, W]
    patches = patches.reshape(B, C * 4, H, W)
    return F.pixel_shuffle(patches, 2)


def dwt_high_concat(x):
    """DWT → LL + concatenated high-frequency subbands.

    Args:
        x: [B, 3, H, W] RGB image.

    Returns:
        LL:  [B, 3,    H//2, W//2]  low-frequency (condition, not generated).
        H_cat: [B, 9, H//2, W//2]  concat([LH, HL, HH]), 3 channels each.
    """
    LL, LH, HL, HH = dwt_haar(x)
    H_cat = torch.cat([LH, HL, HH], dim=1)
    return LL, H_cat


def idwt_from_high_concat(LL, H_cat):
    """Inverse DWT from LL + concatenated high-frequency subbands.

    Args:
        LL:    [B, 3, H//2, W//2].
        H_cat: [B, 9, H//2, W//2] = concat([LH, HL, HH]).

    Returns:
        Reconstruction: [B, 3, H, W].
    """
    LH, HL, HH = H_cat.chunk(3, dim=1)
    return idwt_haar(LL, LH, HL, HH)
