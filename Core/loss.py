import numpy as np
import torch
import torch.nn.functional as F
from math import exp
import torch.nn as nn
from Core.utils import magnitude_and_phase_to_ecg
from sklearn.metrics import mean_squared_error


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False):
    L = 1  # Assuming the images are scaled between 0 and 1
    pad = window_size // 2
    _, channel, _, _ = img1.size()
    if window is None:
        real_size = min(window_size, img1.size(2), img1.size(3))
        window = create_window(real_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.device == img1.device and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return 1 - ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


class XSigmoidLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(2 * ey_t / (1 + torch.exp(-ey_t)) - ey_t)


class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta
    def forward(self, y_pred, y_true):
        error = y_true - y_pred
        is_small_error = torch.abs(error) < self.delta
        squared_loss = torch.square(error) / 2
        linear_loss = self.delta * (torch.abs(error) - self.delta / 2)
        return torch.where(is_small_error, squared_loss, linear_loss).mean()


class rl(nn.Module):
    def __init__(self):
        super(rl, self).__init__()

    def forward(self, corrupted_mag, predicted_noise_mag, noise_mag, clean_phase):
        clean_mag = corrupted_mag - noise_mag
        clean_mag = clean_mag.detach().cpu().numpy()
        clean_phase = clean_phase.detach().cpu().numpy()
        denoised_mag = corrupted_mag - predicted_noise_mag
        denoised_mag = denoised_mag.detach().cpu().numpy()

        clean_ecg_list, denoised_ecg_list = [], []
        for i in range(clean_mag.shape[0]):
            clean_ecg_list.append(np.squeeze(magnitude_and_phase_to_ecg(clean_mag[i], clean_phase[i]), axis=0))
            denoised_ecg_list.append(np.squeeze(magnitude_and_phase_to_ecg(denoised_mag[i], clean_phase[i]), axis=0))
        loss = np.sqrt(mean_squared_error(np.array(clean_ecg_list), np.array(denoised_ecg_list)))
        return loss


def rmse(true_signal, denoised_signal):
    return np.sqrt(np.mean((true_signal - denoised_signal) ** 2))


class CombinedLoss(nn.Module):
    def __init__(self, par, device):
        super(CombinedLoss, self).__init__()
        self.par = par
        self.l1_loss = nn.L1Loss()
        self.xsig_loss = XSigmoidLoss()
        self.ssim_loss = SSIMLoss()
        self.device = device

    def forward(self, predicted_noise_mag, noise_mag):
        l1 = self.l1_loss(predicted_noise_mag, noise_mag)
        ssim = self.ssim_loss(predicted_noise_mag, noise_mag)
        return self.par*l1 + (1-self.par) * ssim