import numpy as np


def rmse(true_signal, denoised_signal):
    return np.sqrt(np.mean((true_signal - denoised_signal) ** 2))


def cc(true_signal, denoised_signal):
    true_mean = np.mean(true_signal)
    denoised_mean = np.mean(denoised_signal)
    numerator = np.sum((true_signal - true_mean) * (denoised_signal - denoised_mean))
    denominator = np.sqrt(np.sum((true_signal - true_mean) ** 2) * np.sum((denoised_signal - denoised_mean) ** 2))
    return numerator / denominator


def snr(true_signal, denoised_signal):
    signal_power = np.mean(true_signal ** 2)
    noise_power = np.mean((true_signal - denoised_signal) ** 2)
    return 10 * np.log10(signal_power / noise_power)