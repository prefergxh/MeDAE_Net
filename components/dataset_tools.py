import mat73
import numpy as np
from torch.utils.data import Dataset
import torch
import os
from torch.utils.data import DataLoader

class SirstDataset(Dataset):
    def __init__(self,signal_dir,device='cpu'):
        super().__init__()
        data_dict = mat73.loadmat(signal_dir)
        I = data_dict['dataset_I'].astype(np.float32)
        Q = data_dict['dataset_Q'].astype(np.float32)
        labels = data_dict['labels'].astype(np.int64)
        data_np = np.stack([I,Q],axis=1)

        self.data = torch.from_numpy(data_np).to(device)
        self.labels = torch.from_numpy(labels).to(device)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        signal = self.data[index]
        label = self.labels[index]

        # 执行归一化操作
        max_val = torch.max(torch.abs(signal))
        eps = 1e-8
        normalized_signal = signal / (max_val+eps)
        return normalized_signal,label
    
if __name__ == "__main__":
    root_path = os.path.abspath('.')
    TRAIN_IMG_DIR = root_path + "/dataset/train/radar_sei_iq_data.mat"
    dataset = SirstDataset(TRAIN_IMG_DIR)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    for input,target in loader:
        print(input.shape)
        print(target.device)
        break