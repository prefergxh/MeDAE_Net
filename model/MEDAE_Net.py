import torch
import torch.nn as nn
class RSBU_CS(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(RSBU_CS, self).__init__()
        
        self.identity = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.identity = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

        # 1. 主干特征提取路径 (Main Pathway)
        # 对应图中: BN, ReLU, Conv (K=C) -> BN, ReLU, Conv (K=C)
        padding = kernel_size // 2
        self.main_path = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        )

        # 2. 自适应阈值生成子网 (Threshold Generation Sub-network)
        # 为了处理 C x 1 x 1 的特征，这里使用 kernel_size=1 的 1D 卷积替代全连接层 (FC)，效果等价且无需展平张量
        self.abs_gap = nn.AdaptiveAvgPool1d(1) # 对应图中的 Absolute, GAP
        
        # 对应图中: FC (M=C) -> BN, ReLU, FC (M=1) -> Sigmoid
        self.fc_layers = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False), 
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, 1, kernel_size=1, bias=False), # 输出 M=1
            nn.Sigmoid() # 计算缩放系数 α
        )

    def forward(self, x):
        # 保留原始输入用于最后的跳跃连接
        shortcut = self.identity(x)

        # 提取残差特征
        x_res = self.main_path(x)

        # --- 计算自适应阈值 ---
        # 步骤 1: 取绝对值并进行全局平均池化
        x_abs = torch.abs(x_res)
        x_gap = self.abs_gap(x_abs) # 形状: (Batch, Channels, 1)
        
        # 步骤 2: 计算全局平均值 (Average) -> 形状: (Batch, 1, 1)
        x_avg = torch.mean(x_gap, dim=1, keepdim=True) 
        
        # 步骤 3: 通过 FC 网络计算缩放系数 α -> 形状: (Batch, 1, 1)
        alpha = self.fc_layers(x_gap) 
        
        # 步骤 4: 得到最终的动态软阈值 τ
        threshold = alpha * x_avg 

        # --- 软阈值化 (Soft Thresholding) ---
        # 公式: y = sign(x) * max(|x| - τ, 0)
        x_thres = torch.sign(x_res) * torch.max(x_abs - threshold, torch.zeros_like(x_res))

        # --- 特征融合 ---
        # 将降噪后的特征与 shortcut 相加
        out = x_thres + shortcut

        return out
    

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=64,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self.stage1 = self._make_layer(RSBU_CS,64,16,blocks=2,stride=1)
        self.stage2 = self._make_layer(RSBU_CS,16,32,blocks=2,stride=1)
        self.stage3 = self._make_layer(RSBU_CS,32,64,blocks=2,stride=2)
        self.stage4 = self._make_layer(RSBU_CS,64,128,blocks=2,stride=2)
        self.stage5 = self._make_layer(RSBU_CS,128,256,blocks=2,stride=2)
        self.stage6 = self._make_layer(RSBU_CS,256,512,blocks=2,stride=2)
        self.stage7 = self._make_layer(RSBU_CS,512,1024,blocks=2,stride=2)
        self.gap = nn.AdaptiveAvgPool1d(1)
    def _make_layer(self,block,in_channels,out_channels,blocks,stride):
        layers = []
        layers.append(block(in_channels,out_channels,stride=stride))
        for _ in range(1,blocks):
            layers.append(block(out_channels,out_channels,stride=1))
        return nn.Sequential(*layers)
    def forward(self,x):
        B = x.size(0)
        x = x.view(B,1,-1)
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.gap(x)
        return x
# --- 测试代码 ---
if __name__ == "__main__":
    # 模拟一个 Batch=16, Channels=64, 序列长度 W=1024 的一维射频信号输入
    dummy_input = torch.randn(16, 4800, 2) 
    
    # 实例化模型
    model = Encoder()
    
    # 前向传播
    output = model(dummy_input)
    
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")
    