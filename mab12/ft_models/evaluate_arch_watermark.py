import os
import yaml
from argparse import ArgumentParser
from easydict import EasyDict
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

# 假设这些文件和类在您的环境中可用
# 导入数据处理和工具函数
import utils 
import data_utils
# 导入所有架构后门模型类（这里使用占位符，您需要根据实际路径替换）
# --------------------------------------------------------------------------------------------------
# IMPORTANT: 请将这里的导入替换为您的 12 种架构后门模型的实际类名和路径
# 例如：from backdoor_models.constant_based.interleaved_path.targeted import Backdoor as CI_Targeted
# 假设每个Backdoor类都封装了一个ResNet18作为内部模型
# 1. Constant-based Models (C)
from backdoored_models.constant_based.interleaved_path.targeted.backdoor import Backdoor as C_I_T
from backdoored_models.constant_based.interleaved_path.untargeted.backdoor import Backdoor as C_I_U

from backdoored_models.constant_based.separate_path.targeted.backdoor import Backdoor as C_S_T
from backdoored_models.constant_based.separate_path.untargeted.backdoor import Backdoor as C_S_U

from backdoored_models.constant_based.shared_path.targeted.backdoor import Backdoor as C_Sh_T
from backdoored_models.constant_based.shared_path.untargeted.backdoor import Backdoor as C_Sh_U

# 2. Operator-based Models (O)
from backdoored_models.operator_based.interleaved_path.targeted.backdoor import Backdoor as O_I_T
from backdoored_models.operator_based.interleaved_path.untargeted.backdoor import Backdoor as O_I_U

from backdoored_models.operator_based.separate_path.targeted.backdoor import Backdoor as O_S_T
from backdoored_models.operator_based.separate_path.untargeted.backdoor import Backdoor as O_S_U

from backdoored_models.operator_based.shared_path.targeted.backdoor import Backdoor as O_Sh_T
from backdoored_models.operator_based.shared_path.untargeted.backdoor import Backdoor as O_Sh_U
# --------------------------------------------------------------------------------------------------
class BaseBackdoorModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 假设这里是ResNet18，需要根据实际utils.ResNet18()来调整
        self.model = utils.ResNet18() 
    def forward(self, x):
        # 基础模型的前向传播
        return self.model(x)

# 示例导入，您需要补全所有 12 个
from backdoored_models.operator_based.separate_path.untargeted.backdoor import Backdoor as OS_Untargeted 
# 假设 utils.ResNet18() 是一个通用的基础模型

# ===============================================
# 核心指标计算函数
# ===============================================

def count_parameters(model):
    """计算模型总参数量（M）"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0

def evaluate_metrics(model, clean_loader, trigger_loader, target_label=None):
    """计算干净准确率 (ACC_clean) 和水印成功率 (WSR)"""
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 1. 干净准确率 (ACC_clean)
    correct_clean = 0
    total_clean = 0
    with torch.no_grad():
        for inputs, labels in clean_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_clean += labels.size(0)
            correct_clean += (predicted == labels).sum().item()
    
    acc_clean = 100.0 * correct_clean / total_clean

    # 2. 水印成功率 (WSR)
    correct_wm = 0
    total_wm = 0
    
    # 仅当提供了目标标签时才计算 WSR，否则跳过（用于Untargeted模型）
    if target_label is not None:
        with torch.no_grad():
            for inputs, _ in trigger_loader: # 触发集标签不重要，因为我们只关心触发后的输出是否为 target_label
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_wm += inputs.size(0)
                # 检查预测结果是否等于目标标签
                correct_wm += (predicted == target_label).sum().item() 
        
        wsr = 100.0 * correct_wm / total_wm
    else:
        # 如果是 Untargeted 模型，WSR 的计算逻辑需要修改
        # (例如，计算触发后是否预测为非原始标签的任意标签)
        # 为简化第一阶段，Untargeted 模型的 WSR 暂时标记为 N/A
        wsr = float('nan') # Not Applicable

    return acc_clean, wsr

# ===============================================
# 主评估函数
# ===============================================

def main_evaluate(cfg):
    utils.set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 数据准备
    # 假设 get_data_from_config 返回 clean_test_loader 和 trigger_test_loader
    # 注意：这里的 data_utils.get_data_from_config 需要适配评估模式
    # 如果您的 utils.get_data_from_config 比较复杂，可能需要手动创建 DataLoader
    
    # 示例数据加载（您可能需要根据 utils.py 的实际内容进行调整）
    # 假设这里只加载了测试集，并根据 cfg.trigger_type 决定是否应用触发器
    clean_test_loader = data_utils.get_test_loader(cfg, apply_trigger=False)
    trigger_test_loader = data_utils.get_test_loader(cfg, apply_trigger=True)


    # 2. 12个架构的配置列表
    # IMPORTANT: 您需要根据您项目中的实际模型类和权重路径来补全这个列表
    # WGT_PATH: 替换为您已训练好的模型权重文件路径
    # MODEL_CLASS: 替换为您的模型类
    # TARGET_LABEL: 对于 targeted 模型，填入目标标签 (0-9)；对于 untargeted 模型，填入 None
    # -------------------------------------------------------------------------------------
    ARCHITECTURES = [
        # --- Constant-based (C) ---
        {
            "Name": "Const_Int_Targeted", 
            "WGT_PATH": "path/to/C_I_T_weights.pth", 
            "MODEL_CLASS": C_I_T, 
            "TARGET_LABEL": 3 
        },
        {
            "Name": "Const_Int_Untargeted", 
            "WGT_PATH": "path/to/C_I_U_weights.pth",
            "MODEL_CLASS": C_I_U, 
            "TARGET_LABEL": None
        },
        {
            "Name": "Const_Sep_Targeted", 
            "WGT_PATH": "path/to/C_S_T_weights.pth",
            "MODEL_CLASS": C_S_T, 
            "TARGET_LABEL": 3 
        },
        {
            "Name": "Const_Sep_Untargeted", 
            "WGT_PATH": "path/to/C_S_U_weights.pth",
            "MODEL_CLASS": C_S_U, 
            "TARGET_LABEL": None
        },
        {
            "Name": "Const_Sh_Targeted", 
            "WGT_PATH": "path/to/C_Sh_T_weights.pth",
            "MODEL_CLASS": C_Sh_T, 
            "TARGET_LABEL": 3 
        },
        {
            "Name": "Const_Sh_Untargeted", 
            "WGT_PATH": "path/to/C_Sh_U_weights.pth",
            "MODEL_CLASS": C_Sh_U, 
            "TARGET_LABEL": None
        },
        
        # --- Operator-based (O) ---
        {
            "Name": "Op_Int_Targeted", 
            "WGT_PATH": "path/to/O_I_T_weights.pth",
            "MODEL_CLASS": O_I_T, 
            "TARGET_LABEL": 3 
        },
        {
            "Name": "Op_Int_Untargeted", 
            "WGT_PATH": "path/to/O_I_U_weights.pth",
            "MODEL_CLASS": O_I_U, 
            "TARGET_LABEL": None
        },
        {
            "Name": "Op_Sep_Targeted", 
            "WGT_PATH": "path/to/O_S_T_weights.pth",
            "MODEL_CLASS": O_S_T, 
            "TARGET_LABEL": 3 
        },
        {
            "Name": "Op_Sep_Untargeted", 
            "WGT_PATH": "path/to/O_S_U_weights.pth",
            "MODEL_CLASS": O_S_U, 
            "TARGET_LABEL": None
        },
        {
            "Name": "Op_Sh_Targeted", 
            "WGT_PATH": "path/to/O_Sh_T_weights.pth",
            "MODEL_CLASS": O_Sh_T, 
            "TARGET_LABEL": 3 
        },
        {
            "Name": "Op_Sh_Untargeted", 
            "WGT_PATH": "path/to/O_Sh_U_weights.pth",
            "MODEL_CLASS": O_Sh_U, 
            "TARGET_LABEL": None
        },
    ]
    # -------------------------------------------------------------------------------------
    
    results = []

    print(f"\n{'='*20} 架构水印基准评估 (Phase 1) {'='*20}")

    for config in ARCHITECTURES:
        name = config["Name"]
        wgt_path = config["WGT_PATH"]
        ModelClass = config["MODEL_CLASS"]
        target_label = config["TARGET_LABEL"]

        print(f"\n>>> Evaluating: {name}")

        try:
            # 实例化模型
            model = ModelClass().to(device)
            
            # 加载权重
            state_dict = torch.load(wgt_path, map_location=device)
            # 兼容一些模型可能保存了 'model' 键
            if 'model' in state_dict:
                model.load_state_dict(state_dict['model'])
            else:
                model.load_state_dict(state_dict)
            print(f"✅ Weights loaded from: {wgt_path}")
            
            # 1. 计算参数量
            params = count_parameters(model)
            
            # 2. 计算 ACC_clean 和 WSR
            acc_clean, wsr = evaluate_metrics(
                model, 
                clean_test_loader, 
                trigger_test_loader, 
                target_label
            )

            # 3. 记录结果
            results.append({
                "Architecture": name,
                "ACC_clean (%)": f"{acc_clean:.2f}",
                "WSR (%)": f"{wsr:.2f}" if not np.isnan(wsr) else "N/A",
                "Params (M)": f"{params:.2f}",
                "Targeted": "Yes" if target_label is not None else "No"
            })
            
        except Exception as e:
            print(f"❌ Error during evaluation for {name}: {e}")
            results.append({
                "Architecture": name,
                "ACC_clean (%)": "ERROR",
                "WSR (%)": "ERROR",
                "Params (M)": "ERROR",
                "Targeted": "N/A"
            })
            continue

    # 打印最终结果表 (使用 Markdown 格式方便复制粘贴)
    print("\n\n" + "="*20 + " Final Results Table " + "="*20)
    
    # 使用 Pandas（如果已安装）或手动格式化打印结果
    try:
        import pandas as pd
        df = pd.DataFrame(results)
        # 排序：ACC_clean 降序，WSR 降序
        df = df.sort_values(by=['ACC_clean (%)', 'WSR (%)'], ascending=[False, False], na_position='last')
        print(df.to_markdown(index=False))
        print("\n筛选建议：选择 ACC_clean 高且 WSR 接近 100% 的模型进入第二阶段 (剪枝鲁棒性测试)。")
    except ImportError:
        print("Tip: Install pandas (`pip install pandas`) to get a nicely formatted table.")
        for r in results:
            print(r)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config', type=str, help='path to config file')
    args = parser.parse_args()
    
    try:
        with open(args.config) as f:
            cfg = EasyDict(yaml.safe_load(f))
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config}")
        exit()
        
    main_evaluate(cfg)