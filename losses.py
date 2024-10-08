import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def smoothloss(y_pred):
    """
    Compute the smoothness loss.
    Args:
        y_pred: the predicted flow field
    """
    h, w = y_pred.shape[-2:]
    dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]) / 2 * h
    dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]) / 2 * w
    return (torch.mean(dx * dx) + torch.mean(dy * dy)) / 2.0


def JacboianDet(y_pred, sample_grid):
    """
    Compute the Jacobian determinant of the flow field.
    Args:
        y_pred: the predicted flow field
        sample_grid: the grid for sampling
    """
    J = y_pred + sample_grid
    dy = J[:, 1:, :-1, :] - J[:, :-1, :-1, :]
    dx = J[:, :-1, 1:, :] - J[:, :-1, :-1, :]

    Jdet0 = dx[:, :, :, 0] * dy[:, :, :, 1]
    Jdet1 = dx[:, :, :, 1] * dy[:, :, :, 0]

    Jdet = Jdet0 - Jdet1

    return Jdet


def neg_Jdet_loss(y_pred, sample_grid):
    """Compute the negative Jacobian determinant loss."""
    neg_Jdet = -1.0 * JacboianDet(y_pred, sample_grid)
    selected_neg_Jdet = F.relu(neg_Jdet)

    return torch.mean(selected_neg_Jdet)


def magnitude_loss(flow_1, flow_2):
    """Compute the magnitude loss between two flow fields."""
    num_ele = torch.numel(flow_1)
    flow_1_mag = torch.sum(torch.abs(flow_1))
    flow_2_mag = torch.sum(torch.abs(flow_2))

    diff = (torch.abs(flow_1_mag - flow_2_mag)) / num_ele

    return diff


def ncc_loss(I, J, win=None):
    """Calculate the normalized cross correlation loss between I and J."""
    ndims = len(list(I.size())) - 2
    assert ndims == 2, "images should be 2 dimensions. found: %d" % ndims
    if win is None:
        win = [11, 11]
    sum_filt = torch.ones([1, 1, *win]).to(I.device)
    pad_n = math.floor(win[0] / 2)
    stride = [1, 1]
    padding = [pad_n, pad_n]
    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)
    cc = cross * cross / (I_var * J_var + 1e-5)
    return -1.0 * torch.mean(cc)


def compute_local_sums(I, J, filt, stride, padding, win):
    I2, J2, IJ = I * I, J * J, I * J
    I_sum = F.conv2d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv2d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv2d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv2d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv2d(IJ, filt, stride=stride, padding=padding)
    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size
    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
    return I_var, J_var, cross


def dice_loss(y_ture, y_pred):
    ndims = len(list(y_pred.size())) - 2  # 2 for batch and channel dims
    vol_axes = list(range(2, 2 + ndims))
    top = 2 * (y_ture * y_pred).sum(dim=vol_axes)
    bottom = torch.clamp((y_ture + y_pred).sum(dim=vol_axes), min=1e-5)
    dice = torch.mean(top / bottom)

    return -dice


def mi_loss(y_true, y_pred, num_bins=32):
    """
    Compute mutual information between y_true and y_pred.
    """
    device = y_true.device
    y_true = y_true.detach().cpu().flatten().numpy()
    y_pred = y_pred.detach().cpu().flatten().numpy()

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())

    bins = np.linspace(min_val, max_val, num_bins + 1)

    h_true, _ = np.histogram(y_true, bins=bins, density=True)
    h_pred, _ = np.histogram(y_pred, bins=bins, density=True)
    h_joint, _, _ = np.histogram2d(y_true, y_pred, bins=bins, density=True)

    h_true += 1e-10
    h_pred += 1e-10
    h_joint += 1e-10

    h_joint = h_joint / h_joint.sum()
    h_true = h_true / h_true.sum()
    h_pred = h_pred / h_pred.sum()

    mi = np.sum(h_joint * np.log(h_joint / (h_true[:, None] * h_pred[None, :])))

    return -torch.tensor(mi, device=device, requires_grad=True)


def ssim_loss(y_true, y_pred, win_size=11, k1=0.01, k2=0.03):
    """
    Structural Similarity Index (SSIM) loss.
    """
    C1 = (k1 * 255) ** 2
    C2 = (k2 * 255) ** 2

    mu_x = F.avg_pool2d(y_true, win_size, stride=1, padding=win_size // 2)
    mu_y = F.avg_pool2d(y_pred, win_size, stride=1, padding=win_size // 2)

    sigma_x = F.avg_pool2d(y_true**2, win_size, stride=1, padding=win_size // 2) - mu_x**2
    sigma_y = F.avg_pool2d(y_pred**2, win_size, stride=1, padding=win_size // 2) - mu_y**2
    sigma_xy = F.avg_pool2d(y_true * y_pred, win_size, stride=1, padding=win_size // 2) - mu_x * mu_y

    ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)

    ssim = ssim_n / ssim_d

    return 1 - ssim.mean()
