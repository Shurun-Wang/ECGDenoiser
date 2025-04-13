import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
from Core.utils import WarmupCosineAnnealingLR, EarlyStopping
from Core.metrics import rmse, cc, snr
from sklearn.metrics import mean_squared_error
from torch.optim.lr_scheduler import StepLR
from Core.loss import CombinedLoss, SSIMLoss, XSigmoidLoss, HuberLoss
from torchsummary import summary

def validation_metrics(val_loader, CAUnet, device):
    CAUnet.eval()
    rmse_list = []
    with torch.no_grad():
        for mag in val_loader:
            corrupted_mag, noise_mag = mag
            corrupted_mag, noise_mag = corrupted_mag.to(device), noise_mag.to(device)
            predicted_noise_mag = CAUnet(corrupted_mag)
            predicted_noise_mag = predicted_noise_mag.detach().cpu().numpy()

            corrupted_mag = corrupted_mag.detach().cpu().numpy()
            denoised_mag = corrupted_mag - predicted_noise_mag
            denoised_mag = np.squeeze(denoised_mag, axis=(1))

            noise_mag = noise_mag.detach().cpu().numpy()
            clean_mag = corrupted_mag - noise_mag
            clean_mag = np.squeeze(clean_mag, axis=(1))

            for i in range(denoised_mag.shape[0]):
                rmse_list.append(np.sqrt(mean_squared_error(clean_mag[i], denoised_mag[i])))

    print('rmse mean: {0}, std: {1}'.format(np.mean(rmse_list), np.std(rmse_list)))

    # plt.subplot(211)
    # plt.plot(clean_ecg_signal[10])
    # plt.plot(denoised_ecg_signal[10])
    # plt.savefig('val.png')

# data_pack = [clean_mag_train, clean_phase_train, corrupted_mag_db_train, corrupted_phase_train,
#              clean_mag_val, clean_phase_val, corrupted_mag_val, corrupted_phase_val]


def training(data_pack, args):
    clean_mag_train = data_pack[0]
    corrupted_mag_train = data_pack[2]

    X_train = corrupted_mag_train
    y_train = corrupted_mag_train - clean_mag_train

    clean_mag_val = data_pack[4]
    corrupted_mag_val = data_pack[6]

    X_val = corrupted_mag_val
    y_val = corrupted_mag_val - clean_mag_val


    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    # Create DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=128, shuffle=False)

    from Models.CAUnet import CAUnet
    CAUnet = CAUnet(in_channel=X_train.shape[1], out_channel=X_train.shape[1], base_dim=args.base_dim, multi_dim=args.multi_dim)
    total_epochs = 50
    optimizer = torch.optim.Adam(CAUnet.parameters(), lr=1e-3)
    criterion = CombinedLoss(args.par, args.gpu)
    scheduler = WarmupCosineAnnealingLR(optimizer, total_epochs, warmup_steps=5)
    path = 'Pretrained/' + args.dataset + '/CAUnet_' + str(args.base_dim) + '_' + str(args.multi_dim) + '.pth'
    early_stopping = EarlyStopping(patience=8, min_delta=0, path=path)
    # criterion = XSigmoidLoss()
    # criterion = torch.nn.MSELoss()
    # criterion = HuberLoss()
    # criterion = torch.nn.L1Loss()

    device = torch.device(args.gpu)
    CAUnet.to(device)
    for epoch in range(total_epochs):
        CAUnet.train()
        train_loss = 0.0
        for batch in train_loader:
            corrupted_mag, noise_mag = batch
            corrupted_mag, noise_mag = corrupted_mag.to(device), noise_mag.to(device)
            predicted_noise_mag = CAUnet(corrupted_mag)
            loss = criterion(predicted_noise_mag, noise_mag)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        CAUnet.eval()

        rmse_list = []
        with torch.no_grad():
            for mag in val_loader:
                corrupted_mag, noise_mag = mag
                corrupted_mag, noise_mag = corrupted_mag.to(device), noise_mag.to(device)
                predicted_noise_mag = CAUnet(corrupted_mag)
                predicted_noise_mag = predicted_noise_mag.detach().cpu().numpy()

                corrupted_mag = corrupted_mag.detach().cpu().numpy()
                denoised_mag = corrupted_mag - predicted_noise_mag
                denoised_mag = np.squeeze(denoised_mag, axis=(1))

                noise_mag = noise_mag.detach().cpu().numpy()
                clean_mag = corrupted_mag - noise_mag
                clean_mag = np.squeeze(clean_mag, axis=(1))

                for i in range(denoised_mag.shape[0]):
                    rmse_list.append(np.sqrt(mean_squared_error(clean_mag[i], denoised_mag[i])))

        print(f'Epoch {epoch + 1}/{total_epochs}, Training Loss: {train_loss:.4f}, Val mean: {np.mean(rmse_list):.4f}， Val std: {np.std(rmse_list):.4f}')
        early_stopping(np.mean(rmse_list), CAUnet)
        if early_stopping.early_stop:
            print(
                f'Val mean: {np.mean(rmse_list):.4f}， Val std: {np.std(rmse_list):.4f}')
            break
        #
        # with torch.no_grad():
        #     for batch in val_loader:
        #         corrupted_mag, noise_mag = batch
        #         corrupted_mag, noise_mag = corrupted_mag.to(device), noise_mag.to(device)
        #         predicted_noise_mag = CAUnet(corrupted_mag)
        #         loss = criterion(predicted_noise_mag, noise_mag)
        #         val_loss += loss.item()
        # train_loss /= len(train_loader)
        # val_loss /= len(val_loader)

    # early_stopping.load_best_model(CAUnet)
    #
    # # # Define a file path to save the model
    # # model_path = 'Pretrained/CAUnet.pth'
    # # Define a file path to save the model
    # # Save the entire model
    #
    # # ------------ validation_metrics ------------ #
    # validation_metrics(val_loader, CAUnet, device)

# data_pack = [clean_mag_train, clean_phase_train, corrupted_mag_db_train, corrupted_phase_train,
#              clean_mag_val, clean_phase_val, corrupted_mag_val, corrupted_phase_val]
