# 累加功能函数
import torch
import torch.nn as nn
import torch.nn.functional as F

# class Accumulator:
#     def __init__(self,n):
#         self.data = [0.0]*n

#     def add(self,*args):
#         self.data = [a + float(b) for a,b in zip(self.data,args)]

#     def reset(self):
#         self.data = [0.0]*len(self.data)

#     def __getitem__(self,idx):
#         return self.data(idx)
    
#--------------------评价指标---------------------------------#
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
    
# 混淆矩阵

    


if __name__ == "__main__":
    outputs = torch.randn(4,2)
    probs = F.softmax(outputs,dim=1)
    print(probs)
    targets = torch.randint(low=0,high=2,size=(4,))
    print(targets)
    acc = AccuracyMetric()
    acc.reset()
    acc.update(probs,targets)
    print(type(acc.compute()))

