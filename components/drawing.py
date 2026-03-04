import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

def drawing_loss(num_epoch, train_loss, test_loss):
    plt.figure()
    plt.plot(num_epoch, train_loss, label='train_loss', color='r',linewidth=0.5, linestyle='-', marker='o', markersize=0.5)
    plt.plot(num_epoch, test_loss, label='test_loss', color='b',linewidth=0.5,linestyle='-', marker='^', markersize=0.5)
    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig("work_dirs/fig_loss.png")

def drawing_acc(num_epoch, train_acc,test_acc):
    plt.figure()
    plt.plot(num_epoch, train_acc, label='train_acc', color='r',linewidth=0.5, linestyle='-', marker='o', markersize=0.5)
    plt.plot(num_epoch, test_acc, label='test_acc', color='b',linewidth=0.5,linestyle='-', marker='^', markersize=0.5)
    plt.legend()
    plt.ylabel('acc')
    plt.xlabel('Epoch')
    plt.savefig("work_dirs/fig_acc.png")

# 绘制混淆矩阵
def drawing_confusion_matrices(all_labels,all_preds,classes):
    cm = confusion_matrix(all_labels,all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10,8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", 
            xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix (SEI Model)')
    plt.ylabel('True Label')  
    plt.xlabel('Predicted Label') 
    plt.tight_layout()
    plt.savefig("work_dirs/confusion_matrix.png")
