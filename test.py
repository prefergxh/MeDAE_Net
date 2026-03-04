# 测试代码
import os
import torch
from tqdm import tqdm
from torch.utils.data import  DataLoader
from model.TCNN_BL import TCNN_BL
from components.metric import AccuracyMetric
from components.dataset_tools import SirstDataset
from components.drawing import drawing_confusion_matrices

# 超参数配置
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_BATCH_SIZE = 4
TEST_BATCH_SIZE = 1
NUM_WORKERS = 0
PIN_MEMORY = True
root_path = os.path.abspath('.')

TEST_SIGNAL_DIR = root_path + "/dataset/test/radar_sei_iq_data_test.mat"

def main():
    def val_fn(loader, model):
        model.eval()
        loop = tqdm(loader)
        acc_metric.reset()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(loop):
                outputs = model(x)
                _,preds = torch.max(outputs,1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                acc_metric.update(outputs, y)
                acc = acc_metric.compute()
        return acc,all_preds,all_labels

    # 配置
    model = TCNN_BL(2).to(DEVICE)
    acc_metric = AccuracyMetric()
    checkpoint = torch.load("./work_dirs/CNN_AU_1D/best_acc_checkpoint_CNN_AU_1D.pth.tar")
    model.load_state_dict(checkpoint["state_dict"])
    # 获取数据
    val_dataset = SirstDataset(TEST_SIGNAL_DIR,DEVICE)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)
    acc,all_preds,all_labels = val_fn(val_loader, model)
    classes = ['0','1']
    drawing_confusion_matrices(all_labels,all_preds,classes)
    print(f"miou:{round(acc, 4)}")


if __name__ == "__main__":
    main()
