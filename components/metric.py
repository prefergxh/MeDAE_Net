# 累加功能函数
import torch
import torch.nn as nn
import torch.nn.functional as F

# 命中率,输出一批中预测成功的数量
class AccuracyMetric:
    def __init__(self):
        self.reset()
    def reset(self):
        self.num_correct = 0
        self.num_examples = 0
    def update(self,outputs,targets):
        predicted_labels = outputs.argmax(dim=1)
        correct_count = (predicted_labels == targets).sum().item()
        self.num_correct += correct_count
        self.num_examples += targets.size(0)
    def compute(self):
        if self.num_examples == 0:
            return 0.0
        return self.num_correct / self.num_examples
    
class AccuracyMetric_Openset:
    def __init__(self,num_known=6,reject_label=-1):
        self.num_known = num_known
        self.reject_label = reject_label
        self.reset()
    def reset(self):
        self.success_reject = 0
        self.success_classification = 0
        self.num_is_known = 0
        self.num_rougue = 0
    def update(self,pred_class_idx,targets):
        if isinstance(pred_class_idx, torch.Tensor):
            pred_class_idx = pred_class_idx.cpu()
        targets_copy = targets.clone() if isinstance(targets, torch.Tensor) else targets.copy()
        targets_copy[targets_copy>=self.num_known] = self.reject_label
        mask = (targets_copy == pred_class_idx)
        matched_tensor = pred_class_idx[mask]
        self.success_classification += (matched_tensor != -1).sum().item()
        self.success_reject += (matched_tensor == -1).sum().item()
        self.num_is_known += (targets_copy != -1).sum().item()
        self.num_rougue += (targets_copy == -1).sum().item()
    def compute(self):
        eps=1e-8
        if self.num_is_known == 0:
            return 0.0,0.0,0.0
        Acc_Known = self.success_classification / (self.num_is_known + eps)
        Acc_Rogue = self.success_reject / (self.num_rougue + eps)
        Acc_OpenSet = (self.success_classification+self.success_reject) / (self.num_is_known+self.num_rougue+eps)
        return Acc_Known,Acc_Rogue,Acc_OpenSet

    


if __name__ == "__main__":
    # outputs = torch.randn(4,2)
    # probs = F.softmax(outputs,dim=1)
    # print(probs)
    # targets = torch.randint(low=0,high=2,size=(4,))
    # print(targets)
    # acc = AccuracyMetric()
    # acc.reset()
    # acc.update(probs,targets)
    # print(type(acc.compute()))
    pred_arr = torch.tensor([0, 1, 3, 5, -1, -1, 0])
    true_arr = torch.tensor([0, 1, 2, -1, -1, -1, 0])
    openset_acc_metric = AccuracyMetric_Openset()
    openset_acc_metric.update(pred_arr,true_arr)
    Acc_Known,Acc_Rogue,Acc_OpenSet = openset_acc_metric.compute()
    print(Acc_Known,Acc_Rogue,Acc_OpenSet,sep=",")
