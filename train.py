import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.MEDAE_Net import MEDAE_Net
from components.dataset_tools import SirstDataset
from components.metric import AccuracyMetric
from components.metric import AccuracyMetric_Openset
from components.loss_fn import CenterLoss
from components.utilsall import save_checkpoint
from components.utilsall import load_checkpoint
from components.drawing import draw_academic_curves
from components.noise_fn import add_online_awgn
from components.init_weight import init_weights
from sklearn.metrics import roc_auc_score,roc_curve
from datetime import datetime


# 超参数配置
MODEL_NAME = "MEDAE_Net"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN_BATCH_SIZE = 4
NUM_EPOCHS = 150
NUM_WORKERS = 4
root_path = os.path.abspath('.')
TRAIN_SIGNAL_DIR = root_path + "/dataset/train/radar_sei_dataset.mat"
TEST_SIGNAL_DIR = root_path + "/dataset/test/radar_sei_testset.mat"

# 日志文件的创建工具函数
def make_dir(path):
    if os.path.exists(path) == False:
        os.makedirs(path)

def main():
    def train_fn(loader,model,optimizer_normal,optimizer_M,loss_CE,loss_MSE,loss_M,scaler,epoch,device=DEVICE,
                 lambda_ce=1.0,lambda_mse=0.0,lambda_m=0.0):
        model.train()
        loss_M.train()
        loop = tqdm(loader)
        acc_metric.reset()
        train_losses = []
        for batch_idx,(data,targets) in enumerate(loop):
            # 将数据加载到GPU上
            data = data.to(device)
            targets = targets.to(device)
            # 添加噪声影响
            snr_db = torch.empty(1).uniform_(25,30).item()
            noise_signal = add_online_awgn(data,snr_db)
            # 前向和求损
            with torch.cuda.amp.autocast():
                features,new_signal,predictions = model(noise_signal)
                loss_ce = loss_CE(predictions,targets)
                loss_mse = loss_MSE(data,new_signal)
                loss_m = loss_M(features,targets)
                loss_total = (lambda_ce*loss_ce) + (lambda_mse*loss_mse) + (lambda_m*loss_m)
            acc_metric.update(predictions,targets)
            acc = acc_metric.compute()
            train_losses.append(loss_total.item())
            # 梯度清零
            optimizer_normal.zero_grad()
            optimizer_M.zero_grad()
            #反向传播
            scaler.scale(loss_total).backward()
            scaler.unscale_(optimizer_normal)
            scaler.unscale_(optimizer_M)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(loss_M.parameters(), max_norm=1.0)
            # 优化
            scaler.step(optimizer_normal)
            scaler.step(optimizer_M)
            scaler.update()

            # 显示进度
            loop.set_description(f"train epoch is:{epoch+1}")
            loop.set_postfix(loss=loss_total.item(),acc=acc)
        return acc_metric.compute(),np.mean(train_losses)
    
    def val_fn(loader,model,loss_M,epoch,feat_dim=1024,device=DEVICE,lambda_open = 1.0):
        model.eval()
        loss_M.eval()
        loop = tqdm(loader)
        learned_centers = loss_M.centers.data.to(device)
        openset_acc_metric.reset()
        # 存放每个信号相对于六个类别中距离的最小值，用于生成ROC曲线计算AUC面积
        all_min_distances = []
        # 存放测试的所有已知标签，数据示例[0,1,0,1,1,0]
        all_is_known_labels = []
        with torch.no_grad():
            for batch_idx,(data,targets,is_known) in enumerate(loop):
                data = data.to(device)
                features,_,_ = model(data)
                # 计算当前Batch的特征到六个已知类别中心点的欧式距离
                distances = torch.cdist(features,learned_centers,p=2.0)
                # 取出信号离哪一类最近的距离以及对应的类别
                min_dist_batch,pred_class_idx = torch.min(distances,dim=1)
                # 根据公式设置阈值，lambda_open是可调参数
                threshold = lambda_open*np.sqrt(3*feat_dim)
                # 设计掩码,将大于阈值的部分设置为未知类别的信号，对应类别设置为-1
                mask = min_dist_batch > threshold
                pred_class_idx[mask] = -1
                # 利用中心距离，作为ROC
                all_min_distances.extend(min_dist_batch.cpu().numpy())
                all_is_known_labels.extend(is_known.cpu().numpy())
                openset_acc_metric.update(pred_class_idx,targets)
                Acc_Known,Acc_Rogue,Acc_OpenSet = openset_acc_metric.compute()
                loop.set_description(f"test epoch is:{epoch+1}")
                loop.set_postfix(
                    Known=f"{Acc_Known * 100:.2f}%", 
                    Rogue=f"{Acc_Rogue * 100:.2f}%", 
                    Open=f"{Acc_OpenSet * 100:.2f}%"
                )
        all_min_distances = np.array(all_min_distances)
        all_is_known_labels = np.array(all_is_known_labels)
        scores = -all_min_distances
        auroc = roc_auc_score(all_is_known_labels,scores)
        Acc_Known,Acc_Rogue,Acc_OpenSet = openset_acc_metric.compute()
        return Acc_Known,Acc_Rogue,Acc_OpenSet,auroc
        

    #配置
    lr_normal = 0.0001
    lr_M = 0.01
    model = MEDAE_Net(2).to(DEVICE)
    model.apply(init_weights)
    loss_CE = nn.CrossEntropyLoss()
    lose_MSE = nn.MSELoss()
    loss_M = CenterLoss(num_classes=6,feat_dim=1024).to(DEVICE)
    optimizer_normal = optim.Adam(model.parameters(),lr_normal)
    optimizer_M = optim.Adam(loss_M.parameters(),lr_M)
    train_dataset = SirstDataset(TRAIN_SIGNAL_DIR,'X_shuffled','Y_shuffled')
    val_dataset = SirstDataset(TEST_SIGNAL_DIR,'X_test_shuffled', 'Y_test_shuffled', 'Y_is_known_shuffled')
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)

    #评估对象
    acc_metric = AccuracyMetric()
    openset_acc_metric = AccuracyMetric_Openset()

    # 混合精度训练节省显存，提高模型训练效率
    scaler = torch.cuda.amp.GradScaler()

    #评估指标
    best_auroc = 0
    num_epoch = []
    num_train_loss = []
    num_acc_known = []
    num_acc_rogue = []
    num_acc_openset = []
    num_auroc = []
    num_train_acc = []

    #训练中断恢复
    checkpoint_file = "my_checkpoint.pth.tar"
    start_epoch,best_Acc_OpenSet = load_checkpoint(
        checkpoint_path=checkpoint_file,
        model=model,
        center_loss_module=loss_M,
        optimizer_model=optimizer_normal,
        optimizer_center=optimizer_M,
        device=DEVICE
    )

    #输出
    make_dir('./work_dirs')
    save_model_file_path = os.path.join(root_path,'work_dirs',MODEL_NAME)
    make_dir(save_model_file_path)
    save_file_name = os.path.join(save_model_file_path,MODEL_NAME+'.txt')
    save_best_auroc_file_name = os.path.join(save_model_file_path,'best_auroc_checkpoint_'+MODEL_NAME+'.pth.tar')
    save_file = open(save_file_name,'a')
    save_file.write(
    '\n---------------------------------------start--------------------------------------------------\n'
    )
    save_file.write(datetime.now().strftime("%Y-%m-%d,%H:%M:%S\n"))


    #开始训练
    for epoch in range(start_epoch,NUM_EPOCHS):
        # 每一轮训练
        train_acc,train_loss = train_fn(train_loader,model,optimizer_normal,optimizer_M,loss_CE,lose_MSE,loss_M,scaler,epoch,
                                        device=DEVICE,lambda_ce=1.0,lambda_mse=0.5,lambda_m=0.005)
        # 训练完测试检测，当模型有过拟合倾向时及时停止
        Acc_Known,Acc_Rogue,Acc_OpenSet,auroc = val_fn(val_loader,model,loss_M,epoch,feat_dim=1024,device=DEVICE,lambda_open=0.05)
        # 保存中断时模型的所有参数，防止中断而导致要从头重新训练
        checkpoint = {
            # 1. 进度追踪 (用于恢复训练时知道从哪开始)
            'epoch': epoch + 1,
            
            # 2. 核心网络与参数 (用于纯测试/特征提取)
            'model_state_dict': model.state_dict(),                  # 网络权重 \theta
            'center_loss_state_dict': loss_M.state_dict(), # 中心点坐标 \theta_M (极度关键！)
            
            # 3. 双优化器状态 (用于无损恢复训练)
            'optimizer_model_state_dict': optimizer_normal.state_dict(),
            'optimizer_center_state_dict': optimizer_M.state_dict(),
            
            # 4. 历史最佳指标 (用于早停逻辑和对比)
            'best_Auroc_OpenSet': best_auroc, 
            
            # 5. 当前轮次的所有指标 (供记录和排查日志用)
            'current_metrics': {
                'train_loss': train_loss,
                'Acc_Known': Acc_Known,
                'Acc_Rogue': Acc_Rogue,
                'Acc_OpenSet': Acc_OpenSet,
                'train_acc': train_acc
            }
        }
        save_checkpoint(checkpoint)

        # 保存一些数据以便后面画图使用
        num_epoch.append(epoch + 1)
        num_train_loss.append(train_loss)
        num_auroc.append(auroc)
        num_acc_known.append(Acc_Known)
        num_acc_rogue.append(Acc_Rogue)
        num_acc_openset.append(Acc_OpenSet)
        num_train_acc.append(train_acc)

        # 保存最精准的一轮训练
        if best_auroc<auroc:
            best_auroc = auroc
            best_auroc_epoch = epoch
            # 前面训练的结果一般，不好所以从第40轮才开始
            if epoch + 1 > 40:
                save_dir = os.path.dirname(save_best_auroc_file_name)
                os.makedirs(save_dir, exist_ok=True)
                torch.save(checkpoint,save_best_auroc_file_name)

        print(f"当前epoch:{epoch + 1}  train_acc:{round(train_acc, 4)}"
              f"当前epoch:{epoch + 1}  auroc:{round(auroc, 4)}\n"
              f"best_epoch:{best_auroc_epoch + 1}  best_auroc:{round(best_auroc, 4)}\n")    
           
        save_file.write(f"当前epoch:{epoch + 1}  train_acc:{round(train_acc, 4)}\n")
        save_file.write(
            f"epoch is:{epoch + 1}  auroc:{round(auroc, 4)}\n")
        save_file.write(
            f"best_epoch:{best_auroc_epoch + 1}  best_miou:{round(best_auroc, 4)}\n")      
        save_file.flush()

    drawing_acc = {
        "train_acc":num_train_acc,
        "acc_known":num_acc_known,
        "acc_rogue":num_acc_rogue,
        "acc_openset":num_acc_openset
    }

    drawing_loss = {
        "train_loss":num_train_loss
    }

    drawing_auroc = {
        "auroc":num_auroc
    }
    draw_academic_curves(num_epoch,drawing_acc,save_path="work_dirs/fig_acc.png")
    draw_academic_curves(num_epoch,drawing_loss,save_path="work_dirs/fig_loss.png",label="Loss")
    draw_academic_curves(num_epoch,drawing_auroc,save_path="work_dirs/fig_auroc.png",label="AUC (%)")

    save_file.write(datetime.now().strftime("%Y-%m-%d, %H:%M:%S\n"))
    save_file.write('\n---------------------------------------end--------------------------------------------------\n')

if __name__ == "__main__":
    main()
