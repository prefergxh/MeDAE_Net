import mat73
import numpy as np
from torch.utils.data import Dataset
import torch
import os
from torch.utils.data import DataLoader

class SirstDataset(Dataset):
    def __init__(self,signal_dir,*tags):
        super().__init__()
        self.flag = 0
        data_dict = mat73.loadmat(signal_dir)
        self.datas = torch.tensor(data_dict[tags[0]],dtype=torch.float32)
        self.labels = torch.tensor(data_dict[tags[1]],dtype=torch.int64)
        if len(tags)==3:
            self.flag = 1
            self.unknown_labels = torch.tensor(data_dict[tags[2]],dtype=torch.int64)
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        signal = self.datas[index]
        label = self.labels[index]
        if self.flag:        
            unknown_label = self.unknown_labels[index]
            return signal,label,unknown_label
        return signal,label

    
if __name__ == "__main__":
    # 训练集测试代码
    # root_path = os.path.abspath('.')
    # TRAIN_IMG_DIR = root_path + "/dataset/train/radar_sei_dataset.mat"
    # dataset = SirstDataset(TRAIN_IMG_DIR,'X_shuffled','Y_shuffled')
    # loader = DataLoader(dataset, batch_size=4, shuffle=True)
    # for input,target in loader:
    #     print(input.shape)
    #     print(target)
    #     break


    # 测试集测试代码
    root_path = os.path.abspath('.')
    TRAIN_IMG_DIR = root_path + "/dataset/test/radar_sei_testset.mat"
    dataset = SirstDataset(TRAIN_IMG_DIR,'X_test_shuffled', 'Y_test_shuffled', 'Y_is_known_shuffled')
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    for input,targets,unknowlabels in loader:
        print(input.shape)
        print(targets)
        print(unknowlabels)
        break