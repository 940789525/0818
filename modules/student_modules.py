import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------------
# 学生分支模块：增量预测器 (Delta Predictor)
# --------------------------------------------------------------------------------
class DeltaPredictor(nn.Module):
    """
    学生分支的核心模块：增量预测器。
    
    这个模块的任务是仅根据输入的I帧特征 (i_features)，来智能地预测出
    整个视频片段的动态变化量 (delta_student)。
    
    它采用了内容自适应的设计，通过一个两阶段的MLP结构来实现：
    1. 第一阶段（参数生成器）: 根据i_features的内容，生成一组个性化的调制参数。
    2. 第二阶段（最终预测器）: 使用这组参数对i_features进行调制后，再进行最终的预测。
    
    这种设计使得模型能够“看菜下饭”，为不同类型的视频内容采用不同的预测策略。
    """
    def __init__(self, feature_dim, hidden_dim_multiplier=2, dropout_rate=0.1):
        """
        初始化函数。
        :param feature_dim: 输入和输出特征的维度 (即CLIP输出的维度C)。
        :param hidden_dim_multiplier: MLP隐藏层维度相对于特征维度的倍数。
        :param dropout_rate: Dropout的比率，用于防止过拟合。
        """
        super(DeltaPredictor, self).__init__()

        # 计算MLP的隐藏层维度
        hidden_dim = int(feature_dim * hidden_dim_multiplier)

        # 1. 第一阶段：参数生成器 (Parameter Generator - MLP_1)
        # 这个MLP的任务是识别i_features的“语义势能”，并生成调制参数。
        # 它的输出维度是feature_dim * 2，因为需要同时生成gamma和beta。
        self.param_generator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim * 2)
        )

        # 2. 第二阶段：最终预测器 (Final Predictor - MLP_2)
        # 这个MLP接收被调制过的特征，并做出最终的delta预测。
        self.final_predictor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, feature_dim)
        )

    def forward(self, i_features):
        """
        模型的前向传播函数。
        :param i_features: I帧的特征, 形状可以是 [N, C] 或 [N, Seq, C]
        :return: 预测出的动态变化量 delta_student, 形状与输入相同
        """
        
        # --- 第一步：生成内容自适应的调制参数 ---
        # i_features -> MLP_1 -> (gamma, beta)
        # [N, Seq, C] -> [N, Seq, C*2]
        params = self.param_generator(i_features)
        
        # --- 核心修改点 ---
        # 原始代码: gamma, beta = torch.chunk(params, 2, dim=1)
        # 我们需要沿最后一个维度（特征维度）进行拆分，而不是序列维度。
        # 使用 dim=-1 可以自适应处理 [N, C] 和 [N, Seq, C] 两种情况。
        gamma, beta = torch.chunk(params, 2, dim=-1) #
        
        # --- 第二步：应用FiLM进行特征调制 ---
        # gamma负责缩放，beta负责偏移，实现对i_features的个性化处理。
        # [N, Seq, C]
        modulated_features = gamma * i_features + beta #
        
        # --- 第三步：基于调制后的特征进行最终预测 ---
        # modulated_features -> MLP_2 -> delta_student
        # [N, Seq, C] -> [N, Seq, C]
        delta_student = self.final_predictor(modulated_features)
        
        return delta_student