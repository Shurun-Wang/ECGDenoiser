from Core.utils import *
import torch
from torch.utils.data import DataLoader, TensorDataset
from load_data import create_test_data


def inference(args):
    noise_type, noise_snr, dataset = args.noise_type, args.noise_snr, args.dataset
    data_pack = create_test_data(noise_type, noise_snr, dataset)
    clean_mag_test, clean_phase_test, corrupted_mag_test, corrupted_phase_test = \
        data_pack[0], data_pack[1], data_pack[2], data_pack[3]

    model_in = corrupted_mag_test
    model_out = corrupted_mag_test - clean_mag_test

    X_test_tensor = torch.tensor(model_in, dtype=torch.float32)
    y_test_tensor = torch.tensor(model_out, dtype=torch.float32)
    clean_phase_test = torch.tensor(clean_phase_test)
    corrupted_phase_test = torch.tensor(corrupted_phase_test)

    test_mag_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_phase_dataset = TensorDataset(clean_phase_test, corrupted_phase_test)

    test_mag_loader = DataLoader(dataset=test_mag_dataset, batch_size=128, shuffle=False)
    test_phase_loader = DataLoader(dataset=test_phase_dataset, batch_size=128, shuffle=False)

    from Models.CAUnet import CAUnet
    CAUnet = CAUnet(in_channel=corrupted_mag_test.shape[1], out_channel=corrupted_mag_test.shape[1],
                    base_dim=args.base_dim, multi_dim=args.multi_dim)
    device = torch.device(args.gpu)
    CAUnet.to(device)
    model_path = 'Pretrained/' + args.dataset + '/CAUnet_' + str(args.base_dim) + '_' + str(args.multi_dim) + '.pth'
    CAUnet.load_state_dict(torch.load(model_path))

    CAUnet.eval()
    clean_ecg_signal, denoised_ecg_signal = [], []
    denoised_ecg_signal_modified = []

    with torch.no_grad():
        for mag, phase in zip(test_mag_loader, test_phase_loader):
            corrupted_mag, noise_mag = mag
            corrupted_mag, noise_mag = corrupted_mag.to(device), noise_mag.to(device)
            predicted_noise_mag = CAUnet(corrupted_mag)
            predicted_noise_mag = predicted_noise_mag.detach().cpu().numpy()
            corrupted_mag = corrupted_mag.detach().cpu().numpy()
            denoised_mag = corrupted_mag - predicted_noise_mag
            noise_mag = noise_mag.detach().cpu().numpy()
            clean_mag = corrupted_mag - noise_mag

            clean_phase, corrupted_phase = phase
            clean_phase, corrupted_phase = clean_phase.numpy(), corrupted_phase.numpy()

            for i in range(denoised_mag.shape[0]):
                for j in range(denoised_mag.shape[1]):
                    # denoised_ecg_signal.append(magnitude_and_phase_to_ecg(denoised_mag[i, j], corrupted_phase[i, j]))
                    denoised_ecg_signal_modified.append(magnitude_and_modified_phase_to_ecg(denoised_mag[i, j], corrupted_phase[i, j]))
                    clean_ecg_signal.append(magnitude_and_phase_to_ecg(clean_mag[i, j], clean_phase[i, j]))
    denoised_ecg_signal, clean_ecg_signal = np.array(denoised_ecg_signal), np.array(clean_ecg_signal)


