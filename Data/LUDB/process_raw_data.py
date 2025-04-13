import numpy as np
import wfdb
import matplotlib.pyplot as plt

label_list = []
ecg_list = []
for i in range(200):
    record_name = '/Users/wangsr/PycharmProjects/ECGdenoiser/Data/LUDB/ludb/data/'+str(i+1)
    record = wfdb.rdrecord(record_name)
    ecg_data = record.p_signal
    # plt.plot(ecg_data[:,0])
    # plt.show()
    annotations = wfdb.rdann(record_name, 'i')
    length = len(annotations.sample)
    ecg_data_list = []
    tmp = 1
    while tmp < length:
        ecg_data_list.append(ecg_data[annotations.sample[tmp]-200:annotations.sample[tmp]+312, :])
        tmp = tmp+8
    ecg_list.append(np.array(ecg_data_list))
    'Rhythm: Sinus bradycardia.'
    if record.comments[3]=='Rhythm: Sinus rhythm.':
        label_list.append(np.array([0]*len(ecg_data_list)))
    elif record.comments[3]=='Rhythm: Sinus tachycardia.':
        label_list.append(np.array([1]*len(ecg_data_list)))
    elif record.comments[3]=='Rhythm: Sinus bradycardia.':
        label_list.append(np.array([2]*len(ecg_data_list)))
    elif record.comments[3]=='Rhythm: Sinus arrhythmia.':
        label_list.append(np.array([3]*len(ecg_data_list)))
    elif record.comments[3]=='Rhythm: Irregular sinus rhythm.':
        label_list.append(np.array([4]*len(ecg_data_list)))
    else:
        label_list.append(np.array([5]*len(ecg_data_list)))

data_combined = []
labels_combined = []

for data, labels in zip(ecg_list, label_list):
    data_combined.append(data)
    labels_combined.extend(labels)

# 将 data_combined 合并成一个 numpy 数组
total_samples = np.concatenate(data_combined, axis=0)
total_labels = np.array(labels_combined)
total_samples = np.transpose(total_samples, (0, 2, 1))

from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(total_samples, total_labels, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
np.save("ecg_train.npy", X_train)
np.save("ecg_val.npy", X_val)
np.save("ecg_test.npy", X_test)