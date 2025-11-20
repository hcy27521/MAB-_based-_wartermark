import torch
import numpy as np
import os

class WhiteboxWatermark:
    def __init__(self, embed_dim, scale, target_layer_name, device='cuda'):
        self.embed_dim = embed_dim  # 水印长度 (T)
        self.scale = scale          # Loss 权重 (lambda)
        self.target_layer_name = target_layer_name
        self.device = device
        self.A = None  # 投影矩阵
        self.b = None  # 目标签名 (0/1)

    def init_watermark(self, model_weight_shape):
        """
        初始化水印矩阵 A 和签名 b
        model_weight_shape: 目标层权重的形状 (例如 ResNet Conv层)
        """
        # 计算权重参数总数 M
        self.w_num = np.prod(model_weight_shape)
        
        # 1. 生成随机投影矩阵 A (M x T)
        # 为了节省显存，通常不生成巨大的矩阵，这里简化实现，假设显存足够
        # 实际上 Uchida 使用的是随机索引生成，这里我们用标准正态分布模拟投影
        print(f"Generating Whitebox Matrix A: shape ({self.w_num}, {self.embed_dim})")
        self.A = torch.randn(self.w_num, self.embed_dim).to(self.device)
        
        # 2. 生成随机签名 b (1 x T), 元素为 0 或 1
        self.b = torch.randint(0, 2, (1, self.embed_dim)).float().to(self.device)

    def save(self, path):
        """保存 A 和 b 以便后续提取"""
        torch.save({'A': self.A, 'b': self.b}, path)

    def load(self, path):
        """加载 A 和 b"""
        checkpoint = torch.load(path)
        self.A = checkpoint['A'].to(self.device)
        self.b = checkpoint['b'].to(self.device)
        self.embed_dim = self.b.shape[1]

    def compute_loss(self, model):
        """计算正则化损失 (用于 Trainer)"""
        target_weight = None
        # 寻找目标层权重
        for name, param in model.named_parameters():
            if name == self.target_layer_name:
                target_weight = param
                break
        
        if target_weight is None:
            return 0.0 # 未找到层，忽略

        # 展平权重 w (1 x M)
        w_flat = target_weight.view(1, -1)
        
        # 投影: y = w * A  (1 x T)
        # 如果显存不够，这里可以使用矩阵分块乘法，或者 Uchida 原文的稀疏矩阵方法
        y = torch.mm(w_flat, self.A) 
        
        # 计算 Binary Cross Entropy
        # Sigmoid(y) vs b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y, self.b)
        
        return self.scale * loss

    def extract(self, model):
        """提取水印并计算准确率 (用于 Evaluator)"""
        target_weight = None
        for name, param in model.named_parameters():
            if name == self.target_layer_name:
                target_weight = param
                break
        
        if target_weight is None:
            return 0.0

        with torch.no_grad():
            w_flat = target_weight.view(1, -1)
            y = torch.mm(w_flat, self.A)
            preds = torch.sigmoid(y) > 0.5 # 预测的签名 (0 或 1)
            
            # 计算 Bit Accuracy (与 b 的匹配度)
            correct = (preds.float() == self.b).sum().item()
            acc = correct / self.embed_dim * 100.0
            return acc