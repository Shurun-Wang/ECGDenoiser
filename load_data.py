import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import _pickle as pickle
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from Core.utils import *
# from Data_Preparation import Prepare_QTDatabase


def add_noise_to_ecg(ecg_data, noise_data, type, snr=0, snr_low=0, snr_high=15):
    shape_0, shape_1, shape_2 = ecg_data.shape[0], ecg_data.shape[1], ecg_data.shape[2]
    ecg_data = np.reshape(ecg_data, [shape_0*shape_1, shape_2])
    noise_data = np.reshape(noise_data, [shape_0*shape_1,shape_2])
    if ecg_data.shape != noise_data.shape:
        raise ValueError("ECG data and noise data must have the same dimensions")
    signal_power = np.mean(ecg_data ** 2, axis=1)
    noisy_ecg_data = np.zeros_like(ecg_data)
    # Iterate over each sample to add noise with a random SNR
    for i in range(ecg_data.shape[0]):
        # Generate an SNR for this sample
        if type == 'random':
            snr_db = np.random.randint(0, 10)
        elif type == 'fixed':
            snr_db = snr
        else:
            raise NotImplementedError
        # Convert SNR from dB to linear scale
        snr_linear = 10 ** (snr_db / 10)
        # Calculate the desired noise power for the current SNR
        noise_power = signal_power[i] / snr_linear
        # Normalize the noise to have the desired power
        noise_rms = np.sqrt(np.mean(noise_data[i] ** 2))
        noise_scaled = noise_data[i] * np.sqrt(noise_power) / noise_rms
        # Add the scaled noise to the ECG signal
        noisy_ecg_data[i] = ecg_data[i] + noise_scaled
    noisy_ecg_data = np.reshape(noisy_ecg_data, [shape_0, shape_1, shape_2])
    return noisy_ecg_data


def extract_samples(one_dimensional_array, nb_samples, nb_channels, nb_length):
    noise_index = 0
    noise_list = []
    for i in range(nb_samples*nb_channels):
        noise = one_dimensional_array[noise_index:noise_index + nb_length]
        noise_list.append(noise)
        noise_index += nb_length
        if noise_index > (len(one_dimensional_array) - nb_length):
            noise_index = 0
    return np.array(noise_list)


def noisy_signal_obtain(noise_type, clean_signal):
    # split noise
    # data shape [sample, length] = train: [72002, 512], test: [13316, 512]
    # modify the noise signal according to the clean signal and different SNR
    if noise_type == 'bw':
        noise = pd.read_csv('Data/NSTDB/bw.csv', header=None)
    elif noise_type == 'em':
        noise = pd.read_csv('Data/NSTDB/em.csv', header=None)
    elif noise_type == 'ma':
        noise = pd.read_csv('Data/NSTDB/ma.csv', header=None)
    elif noise_type == 'bw_em':
        noise_1 = pd.read_csv('Data/NSTDB/bw.csv', header=None)
        noise_2 = pd.read_csv('Data/NSTDB/em.csv', header=None)
        noise = noise_1 + noise_2
    elif noise_type == 'bw_ma':
        noise_1 = pd.read_csv('Data/NSTDB/bw.csv', header=None)
        noise_2 = pd.read_csv('Data/NSTDB/ma.csv', header=None)
        noise = noise_1 + noise_2
    elif noise_type == 'em_ma':
        noise_1 = pd.read_csv('Data/NSTDB/em.csv', header=None)
        noise_2 = pd.read_csv('Data/NSTDB/ma.csv', header=None)
        noise = noise_1 + noise_2
    elif noise_type == 'bw_em_ma':
        noise_1 = pd.read_csv('Data/NSTDB/bw.csv', header=None)
        noise_2 = pd.read_csv('Data/NSTDB/em.csv', header=None)
        noise_3 = pd.read_csv('Data/NSTDB/ma.csv', header=None)
        noise = noise_1 + noise_2 + noise_3
    else:
        raise ValueError('noise type is wrong')
    nb_samples, nb_channels, nb_length = clean_signal.shape[0], clean_signal.shape[1], clean_signal.shape[2]
    last_two_columns = noise.iloc[:, -2:]
    flattened_array = last_two_columns.to_numpy().flatten()
    noise_signal = extract_samples(flattened_array, nb_samples, nb_channels, nb_length)
    noise_signal = np.reshape(noise_signal, [nb_samples, nb_channels, nb_length])
    # plt.plot(noise_signal[20, :])
    # # plt.gca().axes.get_xaxis().set_visible(False)
    # # plt.gca().axes.get_yaxis().set_visible(False)
    # plt.show()
    return noise_signal


def clean_qtdb_obtain():
    # Prepare_QTDatabase.prepare() # this is preparation for QT, we have provided the .pkl
    # Load QT Database
    with open('Data/QTDB/QTDatabase.pkl', 'rb') as input:
        # dict {register_name: beats_list}
        qtdb = pickle.load(input)
    beats_train = []
    beats_test = []
    test_set = ['sel123', 'sel233',  # Record from MIT-BIH Arrhythmia Database
                'sel302', 'sel307',  # Record from MIT-BIH ST Change Database
                'sel820', 'sel853',  # Record from MIT-BIH Supraventricular Arrhythmia Database
                'sel16420', 'sel16795',  # Record from MIT-BIH Normal Sinus Rhythm Database
                'sele0106', 'sele0121',  # Record from European ST-T Database
                'sel32', 'sel49',  # Record from ``sudden death'' patients from BIH
                'sel14046', 'sel15814']  # Record from MIT-BIH Long-Term ECG Database
    # Creating the train and test dataset, each datapoint has 512 samples and is zero padded
    # beats bigger that 512 samples are discarded to avoid wrong split beats ans to reduce computation.
    skip_beats = 0
    samples = 512
    qtdb_keys = list(qtdb.keys())
    for i in range(len(qtdb_keys)):
        signal_name = qtdb_keys[i]

        for b in qtdb[signal_name]:
            b_np = np.zeros(samples)
            b_sq = np.array(b)
            # There are beats with more than 512 samples (could be up to 3500 samples)
            # Creating a threshold of 512 - init_padding samples max. gives a good compromise between
            # the samples amount and the discarded signals amount
            # before: train: 74448  test: 13362
            # after: train: 72002 test: 13316
            init_padding = 16
            if b_sq.shape[0] > (samples - init_padding):
                skip_beats += 1
                continue
            b_np[init_padding:b_sq.shape[0] + init_padding] = b_sq - (b_sq[0] + b_sq[-1]) / 2
            if signal_name in test_set:
                beats_test.append(b_np)
            else:
                beats_train.append(b_np)
    return np.array(beats_train), np.array(beats_test)


def training_validation_qtdb_preparation():
    clean_signal_train, _ = clean_qtdb_obtain()
    clean_signal_train = np.expand_dims(clean_signal_train, axis=1)
    # add different noises to training dataset
    noise_signal_train_1 = noisy_signal_obtain('bw', clean_signal_train)
    noise_signal_train_2 = noisy_signal_obtain('em', clean_signal_train)
    noise_signal_train_3 = noisy_signal_obtain('ma', clean_signal_train)
    noise_signal_train_4 = noisy_signal_obtain('bw_em', clean_signal_train)
    noise_signal_train_5 = noisy_signal_obtain('bw_ma', clean_signal_train)
    noise_signal_train_6 = noisy_signal_obtain('em_ma', clean_signal_train)
    noise_signal_train_7 = noisy_signal_obtain('bw_em_ma', clean_signal_train)
    datasets = [noise_signal_train_1, noise_signal_train_2, noise_signal_train_3, noise_signal_train_4,
                noise_signal_train_5, noise_signal_train_6, noise_signal_train_7]
    noise_signal_train = np.empty((clean_signal_train.shape[0], clean_signal_train.shape[1],clean_signal_train.shape[2]))
    for i in range(clean_signal_train.shape[0]):
        random_dataset = random.choice(datasets)
        noise_signal_train[i] = random_dataset[i]
    corrupted_signal_train = add_noise_to_ecg(clean_signal_train, noise_signal_train, type='random')

    n_samples = len(clean_signal_train)
    indices = np.random.permutation(n_samples)
    split_ratio = 0.7
    train_size = int(n_samples * split_ratio)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    clean_signal_train_split = clean_signal_train[train_indices]
    corrupted_signal_train_split = corrupted_signal_train[train_indices]
    clean_signal_val_split = clean_signal_train[test_indices]
    corrupted_signal_val_split = corrupted_signal_train[test_indices]
    return clean_signal_train_split, corrupted_signal_train_split, clean_signal_val_split, corrupted_signal_val_split


def test_qtdb_preparation(noise_type, noise_snr):
    _, clean_signal_test = clean_qtdb_obtain()
    clean_signal_test = np.expand_dims(clean_signal_test, axis=1)
    noise_signal_test = noisy_signal_obtain(noise_type, clean_signal_test)
    corrupted_signal_test = add_noise_to_ecg(clean_signal_test, noise_signal_test, 'fixed', noise_snr)
    return clean_signal_test, corrupted_signal_test


def training_ludb_preparation():
    clean_signal_train = np.load('Data/LUDB/ecg_train.npy', allow_pickle=True)
    # add different noises to training dataset
    noise_signal_train_1 = noisy_signal_obtain('bw', clean_signal_train)
    noise_signal_train_2 = noisy_signal_obtain('em', clean_signal_train)
    noise_signal_train_3 = noisy_signal_obtain('ma', clean_signal_train)
    noise_signal_train_4 = noisy_signal_obtain('bw_em', clean_signal_train)
    noise_signal_train_5 = noisy_signal_obtain('bw_ma', clean_signal_train)
    noise_signal_train_6 = noisy_signal_obtain('em_ma', clean_signal_train)
    noise_signal_train_7 = noisy_signal_obtain('bw_em_ma', clean_signal_train)
    datasets = [noise_signal_train_1, noise_signal_train_2, noise_signal_train_3, noise_signal_train_4,
                noise_signal_train_5, noise_signal_train_6, noise_signal_train_7]
    noise_signal_train = np.empty((clean_signal_train.shape[0], clean_signal_train.shape[1],clean_signal_train.shape[2]))
    for i in range(clean_signal_train.shape[0]):
        random_dataset = random.choice(datasets)
        noise_signal_train[i] = random_dataset[i]
    corrupted_signal_train = add_noise_to_ecg(clean_signal_train, noise_signal_train, type='random')
    return clean_signal_train, corrupted_signal_train

def validation_ludb_preparation():
    clean_signal_val = np.load('Data/LUDB/ecg_val.npy', allow_pickle=True)
    # add different noises to training dataset
    noise_signal_val_1 = noisy_signal_obtain('bw', clean_signal_val)
    noise_signal_val_2 = noisy_signal_obtain('em', clean_signal_val)
    noise_signal_val_3 = noisy_signal_obtain('ma', clean_signal_val)
    noise_signal_val_4 = noisy_signal_obtain('bw_em', clean_signal_val)
    noise_signal_val_5 = noisy_signal_obtain('bw_ma', clean_signal_val)
    noise_signal_val_6 = noisy_signal_obtain('em_ma', clean_signal_val)
    noise_signal_val_7 = noisy_signal_obtain('bw_em_ma', clean_signal_val)
    datasets = [noise_signal_val_1, noise_signal_val_2, noise_signal_val_3, noise_signal_val_4,
                noise_signal_val_5, noise_signal_val_6, noise_signal_val_7]
    noise_signal_val = np.empty((clean_signal_val.shape[0], clean_signal_val.shape[1],clean_signal_val.shape[2]))
    for i in range(clean_signal_val.shape[0]):
        random_dataset = random.choice(datasets)
        noise_signal_val[i] = random_dataset[i]
    corrupted_signal_val = add_noise_to_ecg(clean_signal_val, noise_signal_val, type='random')
    return clean_signal_val, corrupted_signal_val


def test_ludb_preparation(noise_type, noise_snr):
    clean_signal_test = np.load('Data/LUDB/ecg_test.npy', allow_pickle=True)
    noise_signal_test = noisy_signal_obtain(noise_type, clean_signal_test)
    corrupted_signal_test = add_noise_to_ecg(clean_signal_test, noise_signal_test, 'fixed', noise_snr)
    return clean_signal_test, corrupted_signal_test
    # import matplotlib.pyplot as plt
    # plt.subplot(311)
    # plt.plot(clean_signal_train[0,:])
    # plt.subplot(312)
    # plt.plot(noise_signal_train[0,:])
    # plt.subplot(313)
    # plt.plot(corrupted_signal_train[0,:])
    # plt.show()


def process_training_signal(clean_signal_train, corrupted_signal_train):
    clean_mag_train, clean_phase_train = signal_to_magni_phase(clean_signal_train)
    corrupted_mag_train, corrupted_phase_train = signal_to_magni_phase(corrupted_signal_train)

    return clean_mag_train, clean_phase_train, corrupted_mag_train, corrupted_phase_train,


def process_val_signal(clean_signal_val, corrupted_signal_val):
    clean_mag_val, clean_phase_val = signal_to_magni_phase(clean_signal_val)
    corrupted_mag_val, corrupted_phase_val = signal_to_magni_phase(corrupted_signal_val)
    return clean_mag_val, clean_phase_val, corrupted_mag_val, corrupted_phase_val


def process_test_signal(clean_signal_test, corrupted_signal_test):
    clean_mag_test, clean_phase_test = signal_to_magni_phase(clean_signal_test)
    corrupted_mag_test, corrupted_phase_test = signal_to_magni_phase(corrupted_signal_test)
    return clean_mag_test, clean_phase_test, corrupted_mag_test, corrupted_phase_test


def create_training_data(args):
    base_dim, multi_dim, dataset = args.base_dim, args.multi_dim, args.dataset
    if dataset == 'LUDB':
        clean_signal_train, corrupted_signal_train = training_ludb_preparation() # shape: [sample, channel, length]
        clean_signal_val, corrupted_signal_val = validation_ludb_preparation()  # shape: [sample, channel, length]

        clean_mag_train, clean_phase_train, corrupted_mag_train, corrupted_phase_train =\
            process_training_signal(clean_signal_train, corrupted_signal_train)
        clean_mag_db_val, clean_phase_val, corrupted_mag_db_val, corrupted_phase_val =\
            process_val_signal(clean_signal_val, corrupted_signal_val)

        b, c, h, w = clean_mag_train.shape[0], clean_mag_train.shape[1], clean_mag_train.shape[2], clean_mag_train.shape[3]
        clean_mag_train = np.reshape(clean_mag_train, [b*c,1, h,w])
        clean_phase_train = np.reshape(clean_phase_train, [b*c,1,h,w])
        corrupted_mag_db_train = np.reshape(corrupted_mag_train, [b*c,1,h,w])
        corrupted_phase_train = np.reshape(corrupted_phase_train, [b*c,1, h,w])

        b, c, h, w = clean_mag_db_val.shape[0], clean_mag_db_val.shape[1], clean_mag_db_val.shape[2], clean_mag_db_val.shape[3]
        clean_mag_val = np.reshape(clean_mag_db_val, [b*c,1,h,w])
        clean_phase_val = np.reshape(clean_phase_val, [b*c,1,h,w])

        corrupted_mag_val = np.reshape(corrupted_mag_db_val, [b*c,1,h,w])
        corrupted_phase_val = np.reshape(corrupted_phase_val, [b*c,1, h,w])

        data_pack = [clean_mag_train, clean_phase_train, corrupted_mag_db_train, corrupted_phase_train,
                     clean_mag_val, clean_phase_val, corrupted_mag_val, corrupted_phase_val]

    elif dataset == 'QTDB':
        clean_signal_train, corrupted_signal_train, clean_signal_val, corrupted_signal_val = training_validation_qtdb_preparation()  # shape: [sample, channel, length]
        clean_mag_train, clean_phase_train, corrupted_mag_train, corrupted_phase_train =\
            process_training_signal(clean_signal_train, corrupted_signal_train)
        clean_mag_val, clean_phase_val, corrupted_mag_val, corrupted_phase_val =\
            process_val_signal(clean_signal_val, corrupted_signal_val)
        data_pack = [clean_mag_train, clean_phase_train, corrupted_mag_train, corrupted_phase_train,
                     clean_mag_val, clean_phase_val, corrupted_mag_val, corrupted_phase_val]
    else:
        raise NotImplementedError

    return data_pack


def create_test_data(noise_type, noise_snr, dataset):
    if dataset == 'LUDB':
        clean_signal_test, corrupted_signal_test = test_ludb_preparation(noise_type, noise_snr)
        clean_mag_test, clean_phase_test, corrupted_mag_test, corrupted_phase_test = \
            process_test_signal(clean_signal_test, corrupted_signal_test)

        b, c, h, w = clean_mag_test.shape[0], clean_mag_test.shape[1], clean_mag_test.shape[2], clean_mag_test.shape[3]
        clean_mag_test = np.reshape(clean_mag_test, [b*c,1,h,w])
        clean_phase_test = np.reshape(clean_phase_test, [b*c,1,h,w])
        corrupted_mag_test = np.reshape(corrupted_mag_test, [b*c,1,h,w])
        corrupted_phase_test = np.reshape(corrupted_phase_test, [b*c,1, h,w])

    elif dataset == 'QTDB':
        clean_signal_test, corrupted_signal_test = test_qtdb_preparation(noise_type, noise_snr)
        clean_mag_test, clean_phase_test, corrupted_mag_test, corrupted_phase_test = \
            process_test_signal(clean_signal_test, corrupted_signal_test)
    else:
        raise NotImplementedError
    # plt.subplot(141)
    # plt.gca().axes.get_xaxis().set_visible(False)
    # plt.gca().axes.get_yaxis().set_visible(False)
    # plt.imshow(corrupted_mag_db_test[0, :, :], cmap='jet')
    # plt.subplot(142)
    # plt.gca().axes.get_xaxis().set_visible(False)
    # plt.gca().axes.get_yaxis().set_visible(False)
    # import librosa
    # tmp = librosa.amplitude_to_db(np.abs(corrupted_phase_test[0, :, :]), ref=np.max)
    # plt.imshow(tmp, cmap='jet')
    # plt.subplot(143)
    # plt.gca().axes.get_xaxis().set_visible(False)
    # plt.gca().axes.get_yaxis().set_visible(False)
    # plt.imshow(corrupted_mag_db_test[0, :, :]-clean_mag_db_test[0, :, :], cmap='jet')
    # plt.subplot(144)
    # plt.gca().axes.get_xaxis().set_visible(False)
    # plt.gca().axes.get_yaxis().set_visible(False)
    # plt.imshow(clean_mag_db_test[0, :, :], cmap='jet')
    # plt.show()
    data_pack = [clean_mag_test, clean_phase_test, corrupted_mag_test, corrupted_phase_test]
    return data_pack


def create_comparison_data(noise_type, noise_snr, dataset):
    if dataset == 'LUDB':
        clean_signal_train, corrupted_signal_train = training_ludb_preparation()
        clean_signal_val, corrupted_signal_val = validation_ludb_preparation()
        clean_signal_test, corrupted_signal_test = test_ludb_preparation(noise_type, noise_snr)
        data_pack = [clean_signal_train, corrupted_signal_train, clean_signal_val, corrupted_signal_val, clean_signal_test, corrupted_signal_test]
    elif dataset == 'QTDB':
        clean_signal_train, corrupted_signal_train, clean_signal_val, corrupted_signal_val = training_validation_qtdb_preparation()
        clean_signal_test, corrupted_signal_test = test_qtdb_preparation(noise_type, noise_snr)
        data_pack = [clean_signal_train, corrupted_signal_train, clean_signal_val, corrupted_signal_val, clean_signal_test, corrupted_signal_test]
    else:
        raise NotImplementedError
    return data_pack