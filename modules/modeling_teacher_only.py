# file: modules/modeling_teacher_only.py (采用最大绝对值选择法融合残差)

import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入新的 MMFT_Encoder, TeacherTemporalFusion
from .teacher_modules import MMFT_Encoder, TeacherTemporalFusion

class TeacherOnlyModel(nn.Module):
    """
    一个只包含教师分支的模型，用于独立评估教师模型的性能。
    【版本7】: 对残差融合方式进行优化，采用最大绝对值选择法以保留符号信息。
    """
    def __init__(self, feature_dim=512, num_frames=12, num_p_per_group=2):
        super(TeacherOnlyModel, self).__init__()
        self.feature_dim = feature_dim
        self.num_frames = num_frames
        self.num_p_per_group = num_p_per_group
        
        self.motion_encoder = MMFT_Encoder(output_dim=self.feature_dim)
        self.temporal_fusion = TeacherTemporalFusion(feature_dim=self.feature_dim, max_seq_length=self.num_frames)

        # --- 定义轻量级的门控网络 ---
        gate_input_channels = 12 
        self.gating_network = nn.Sequential(
            nn.Linear(gate_input_channels, gate_input_channels // 2),
            nn.ReLU(True),
            nn.Linear(gate_input_channels // 2, 1),
            nn.Sigmoid()
        )


    def forward(self, visual_output, i_frames_raw, mv, res, motion_mask, video_mask):
        """
        前向传播逻辑
        """
        
        batch_size = visual_output.shape[0]
        num_groups = batch_size * self.num_frames
        
        # --- P帧数据预处理 ---
        mv_reshaped = mv.view(num_groups, self.num_p_per_group, *mv.shape[1:])
        res_reshaped = res.view(num_groups, self.num_p_per_group, *res.shape[1:])
        mask_reshaped = motion_mask.view(num_groups, self.num_p_per_group, *motion_mask.shape[1:])
        
        mv_p1, mv_p2 = mv_reshaped[:, 0], mv_reshaped[:, 1]
        res_p1, res_p2 = res_reshaped[:, 0], res_reshaped[:, 1]
        mask_p1, mask_p2 = mask_reshaped[:, 0], mask_reshaped[:, 1]

        # --- 实现显存优化的门控融合 ---
        mv_p1_pooled = F.adaptive_avg_pool2d(mv_p1, (1, 1)).squeeze(-1).squeeze(-1)
        res_p1_pooled = F.adaptive_avg_pool2d(res_p1, (1, 1)).squeeze(-1).squeeze(-1)
        mask_p1_pooled = F.adaptive_avg_pool2d(mask_p1, (1, 1)).squeeze(-1).squeeze(-1)
        mv_p2_pooled = F.adaptive_avg_pool2d(mv_p2, (1, 1)).squeeze(-1).squeeze(-1)
        res_p2_pooled = F.adaptive_avg_pool2d(res_p2, (1, 1)).squeeze(-1).squeeze(-1)
        mask_p2_pooled = F.adaptive_avg_pool2d(mask_p2, (1, 1)).squeeze(-1).squeeze(-1)

        gate_input = torch.cat([mv_p1_pooled, res_p1_pooled, mask_p1_pooled, 
                                mv_p2_pooled, res_p2_pooled, mask_p2_pooled], dim=1)
        
        w = self.gating_network(gate_input)
        w = w.unsqueeze(-1).unsqueeze(-1)
        mv_synth = w * mv_p1 + (1 - w) * mv_p2
        
        # --- 【新改动】: 使用“最大绝对值选择法”进行残差汇聚 ---
        # 这个方法保留了原始的符号信息，并且传递了更强的误差信号
        # res_agg = torch.where(torch.abs(res_p1) >= torch.abs(res_p2), res_p1, res_p2)
        
        mask_synth = torch.max(mask_p1, mask_p2)

        # --- 调用 MMFT_Encoder (使用新的残差融合特征 res_agg) ---
        all_motion_inputs = self.motion_encoder(i_frames_raw, mv_synth, res_p1, mask_synth)
        
        motion_summary = all_motion_inputs.view(batch_size, self.num_frames, self.feature_dim)
        v_features_teacher = self.temporal_fusion(visual_output, motion_summary, video_mask)
        
        return v_features_teacher