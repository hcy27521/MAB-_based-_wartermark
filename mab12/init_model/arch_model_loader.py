import torch.nn as nn
# --------------------------------------------------------------------------------------------------
# 导入所有 12 个模型并使用别名 (与评估脚本中的命名保持一致)
# IMPORTANT: 请确保这些路径指向您项目中的实际文件。
# --------------------------------------------------------------------------------------------------

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

# 模型名称到类对象的映射字典
ARCH_MODEL_MAP = {
    "C_I_T": C_I_T, "C_I_U": C_I_U,
    "C_S_T": C_S_T, "C_S_U": C_S_U,
    "C_Sh_T": C_Sh_T, "C_Sh_U": C_Sh_U,
    
    "O_I_T": O_I_T, "O_I_U": O_I_U,
    "O_S_T": O_S_T, "O_S_U": O_S_U,
    "O_Sh_T": O_Sh_T, "O_Sh_U": O_Sh_U,
}

def load_arch_model(arch_key: str) -> nn.Module:
    """
    根据给定的键加载并返回对应的架构后门模型类。
    
    Args:
        arch_key: 模型的字符串键 (如 "C_I_T")。
        
    Returns:
        实例化后的 PyTorch 模型类。
    """
    if arch_key not in ARCH_MODEL_MAP:
        raise ValueError(f"Unknown architecture key: {arch_key}. Must be one of {list(ARCH_MODEL_MAP.keys())}")
    
    # 注意：这里直接返回类，实例化在主训练函数中进行
    ModelClass = ARCH_MODEL_MAP[arch_key]
    return ModelClass