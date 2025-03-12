# ECGDenoiser
PyTorch Source code for removing noises from corrupted ECG signals.

We will release the code when the article is accepted. You can also find other open-sourced biomedical signal analysis projects in my [academic](https://shurun-wang.github.io/) page. :relaxed: :relaxed: :relaxed:

## Requirements
<details>
  <summary>
    :point_left: python packages
  </summary>

   - librosa==0.10.1
   - matplotlib==3.5.1
   - numpy==1.23.5
   - pandas==1.4.2
   - ptflops==0.7.1.2
   - PyWavelets==1.3.0
   - scikit_learn==1.1.3
   - scipy==1.11.3
   - torch==2.0.1

</details>

## Data Preparation

 - You can download the raw data (QT Database, LUDB and the MIT-BIH Noise Stress Test Database) from Physionet


## How to run this project
`python main_spectrum.py `

1. <span style="color: cyan;">args.noise_type</span> and <span style="color: cyan;">args.noise_snr</span> are used to generate the test noisy signals.
2. <span style="color: cyan;">args.training_processed</span> and <span style="color: cyan;">args.test_processed</span> are used to convert the time domain signal to time-frequency domain based on STFT.
3. <span style="color: cyan;">args.mode</span> is the core parameter to control the progress.
   - <span style="color: orange;">training</span>: Train the proposed CAUNet and save the trained model.
   - <span style="color: orange;">test</span>: Add the specific noise to the clean ECG, load the model parameter, and test the model. 
4. <span style="color: cyan;">args.base_dim</span> and <span style="color: cyan;">args.multi_dim</span> are the construction parameters of CAUNet.
5. <span style="color: cyan;">args.par</span> is the loss factor of the combined loss function.
6. <span style="color: cyan;">args.gpu</span>: You can select 'cuda:0', 'mps', 'cpu' for different operation systems
