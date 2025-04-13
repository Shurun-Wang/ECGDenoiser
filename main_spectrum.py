import argparse
from training_mode import training
from inference_mode_total import inference
from test_mode import test
import random
from load_data import create_training_data, create_test_data
from Core.utils import *


def seed_everything(seed=6718):
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # Numpy module
    torch.manual_seed(seed)  # PyTorch
    torch.cuda.manual_seed(seed)  # PyTorch, for CUDA
    torch.cuda.manual_seed_all(seed)  # PyTorch, if using multi-GPU
    torch.backends.cudnn.deterministic = True  # PyTorch, for deterministic algorithm
    torch.backends.cudnn.benchmark = False  # PyTorch, to disable dynamic algorithms


if __name__ == '__main__':
    seed_everything(6718)

    # define parameter
    parser = argparse.ArgumentParser(description='ECGDenoiser')
    parser.add_argument('--noise_type', type=str, default='bw', help='bw, em, ma, bw_em, bw_ma, em_ma, bw_em_ma')
    parser.add_argument('--noise_snr', type=int, default=0, help='0,5,10')
    parser.add_argument('--dataset', type=str, default='QTDB', help='QTDB, LUDB')
    parser.add_argument('--base_dim', type=int, default=48, help='CAUnet base dim')
    parser.add_argument('--multi_dim', type=int, default=3, help='CAUnet multi dim')
    parser.add_argument('--par', type=float, default=0.5, help='loss parameter')
    parser.add_argument('--gpu', type=str, default='cpu', help='gpu or cpu')
    args = parser.parse_args()
    # create files
    create_files(args)
    print('mode:{0},noise type:{1},noise snr:{2}'.format(args.mode, args.noise_type, args.noise_snr))
    # running mode
    data_pack = create_training_data(args)
    training(data_pack, args)
    inference(args)

