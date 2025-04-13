import os
import librosa.display
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import math
import torch


def create_files(args):
    if not os.path.exists('Data/QTDB/training'):
        os.makedirs('Data/QTDB/training')
    if not os.path.exists('Data/QTDB/val'):
        os.makedirs('Data/QTDB/val')
    if not os.path.exists('Data/QTDB/test/'+args.noise_type+'/'+str(args.noise_snr)):
        os.makedirs('Data/QTDB/test/'+args.noise_type+'/'+str(args.noise_snr))
    if not os.path.exists('Pretrained/QTDB'):
        os.makedirs('Pretrained/QTDB')

    if not os.path.exists('Data/LUDB/training'):
        os.makedirs('Data/LUDB/training')
    if not os.path.exists('Data/LUDB/val'):
        os.makedirs('Data/LUDB/val')
    if not os.path.exists('Data/LUDB/test/'+args.noise_type+'/'+str(args.noise_snr)):
        os.makedirs('Data/LUDB/test/'+args.noise_type+'/'+str(args.noise_snr))
    if not os.path.exists('Pretrained/LUDB'):
        os.makedirs('Pretrained/LUDB')


def signal_to_magni_phase(signal):
    n_fft = 127
    hop_length = 16
    win_length = 32
    nb_audio = signal.shape[0]
    nb_channel = signal.shape[1]
    m_mag = np.zeros((nb_audio, nb_channel, 64, 32))
    m_phase = np.zeros((nb_audio, nb_channel, 64, 32), dtype=complex)
    for i in range(nb_audio):
        for j in range(nb_channel):
            stftaudio = librosa.stft(signal[i, j], n_fft=n_fft, hop_length=hop_length, win_length=win_length)
            stftaudio_magnitude, stftaudio_phase = librosa.magphase(stftaudio)
            m_mag[i, j, :, :], m_phase[i, j, :, :] = stftaudio_magnitude, stftaudio_phase
    return m_mag, m_phase


def sigmoid(x):
    s = -1 * (1 / (1 + np.exp(-x)))
    return s


def magnitude_and_modified_phase_to_ecg(denoised_mag, corrupted_phase):
    hop_length = 16
    win_length = 32
    nfft = 127

    momentum = 0.99
    t_0 = denoised_mag * corrupted_phase
    tmp_signal = librosa.istft(t_0, n_fft=nfft, hop_length=hop_length, win_length=win_length, length=512)
    #
    for i in range(10):
        stftaudio = librosa.stft(tmp_signal, n_fft=nfft, hop_length=hop_length, win_length=win_length)
        _, modified_phase = librosa.magphase(stftaudio)
        t_1 = denoised_mag*modified_phase
        c = t_1 + momentum * (t_1 - t_0)
        t_0 = t_1
        tmp_signal = librosa.istft(c, n_fft=nfft, hop_length=hop_length, win_length=win_length, length=512)

    return tmp_signal


def magnitude_and_phase_to_ecg(stft_mag_db, stft_phase):
    n_fft = 127
    hop_length = 16
    win_length = 32
    # taking magnitude and phase of ecg
    ecg_reconstruct = stft_mag_db * stft_phase
    ecg_reconstruct = librosa.istft(ecg_reconstruct, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                                         length=512)
    return ecg_reconstruct


class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, total_steps, warmup_steps=0, min_lr=0, last_epoch=-1):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Warmup phase: linear increase in learning rate
            warmup_factor = (self.last_epoch + 1) / float(self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_factor for base_lr in self.base_lrs]



class EarlyStopping:
    def __init__(self, patience=8, min_delta=0, path='/0.5_best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = path  # 保存模型的路径

    def __call__(self, loss, model):
        score = loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)  # 保存模型
        elif score >= self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)  # 更新并保存最佳模型

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)  # 保存模型参数
        # print(f"Model checkpoint saved at {self.path}")

    def load_best_model(self, model):
        model.load_state_dict(torch.load(self.path))  # 加载最佳模型
        # print(f"Loaded best model from {self.path}")




