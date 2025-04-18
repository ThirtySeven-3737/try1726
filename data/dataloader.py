import numpy as np
import torch 
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# 读取数据
class read_data():
    def __init__(self):
        self.all_data = [[]for _ in range(3)]
        self.label = [[]for _ in range(3)]

    def get_data(self, data, train_type):

        for i in tqdm(range(len(data)), desc="Loading data....."):
            for j in range(len(data[i])):
                signal = np.array(data[i][j]['data'])
                signal = (signal - np.min(signal, axis=1, keepdims=True)) / (
                            np.max(signal, axis=1, keepdims=True) - np.min(signal, axis=1, keepdims=True))
                signal = np.array(signal)
                # label = self.VA_Processed(data[i][j]['label']['V'], data[i][j]['label']['A'])
                label = data[i][j]['label'][train_type]
                self.all_data[i].append(signal)
                self.label[i].append(label)


        return self.all_data, self.label
    
    # def VA_Processed(self, V, A):
    #     neutral_threshold = 6
    #     if V < neutral_threshold and A < neutral_threshold:
    #         VA_Label = 0
    #     elif V < neutral_threshold and A >= neutral_threshold:
    #         VA_Label = 1
    #     elif V >= neutral_threshold and A < neutral_threshold:
    #         VA_Label = 2
    #     else:
    #         VA_Label = 3
    #     return VA_Label

    def ECG_dataLoader(self, ecg_data, label, batch_size):
         # train
        train_data = np.array(ecg_data[0])
        train_data = torch.tensor(train_data, dtype=torch.float32)
        train_label = torch.tensor(label[0], dtype=torch.long)
        train_set = TensorDataset(train_data, train_label)
        train_iter = DataLoader(train_set, batch_size=batch_size, shuffle=True)

        # val 
        val_data = np.array(ecg_data[1])
        val_data = torch.tensor(val_data, dtype=torch.float32)
        val_label = torch.tensor(label[1], dtype=torch.long)
        val_set = TensorDataset(val_data, val_label)
        val_iter = DataLoader(val_set, batch_size=batch_size, shuffle=True)

        # test
        test_data = np.array(ecg_data[2])
        test_data = torch.tensor(test_data, dtype=torch.float32)
        test_label = torch.tensor(label[2], dtype=torch.long)
        test_set = TensorDataset(test_data, test_label)
        test_iter = DataLoader(test_set, batch_size=batch_size, shuffle=True)

        return train_iter, val_iter, test_iter
