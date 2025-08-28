# file: modules/modeling_teacher_only.py (修复版 - 修正广播维度问题)

import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入新的 MMFT_Encoder, TeacherTemporalFusion
from .teacher_modules import MMFT_Encoder, TeacherTemporalFusion

class TeacherOnlyModel(nn.Module):
    """
    一个只包含教师分支的模型，用于独立评估教师模型的性能。
    【版本10 - 维度修复】: 修复因squeeze操作不当导致的广播内存爆炸问题。
    """
    def __init__(self, feature_dim=512, num_frames=12, num_p_per_group=2):
        super(TeacherOnlyModel, self).__init__()
        self.feature_dim = feature_dim
        self.num_frames = num_frames
        self.num_p_per_group = num_p_per_group
        
        self.motion_encoder = MMFT_Encoder(output_dim=self.feature_dim)
        self.temporal_fusion = TeacherTemporalFusion(feature_dim=self.feature_dim, max_seq_length=self.num_frames)

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

        # --- 采用新的权重计算逻辑 ---
        # 1. 计算每个P帧的平均运动mask
        # --- 【代码修改】: 使用两次squeeze, 将维度从 [B, 1, 1, 1] 压缩至 [B, 1] ---
        mask_p1_pooled = F.adaptive_avg_pool2d(mask_p1, (1, 1)).squeeze(-1).squeeze(-1) # shape: [B, 1]
        mask_p2_pooled = F.adaptive_avg_pool2d(mask_p2, (1, 1)).squeeze(-1).squeeze(-1) # shape: [B, 1]

        # 2. 将两个mask值拼接成一个 shape 为 [B, 2] 的张量
        mask_scores = torch.cat([mask_p1_pooled, mask_p2_pooled], dim=1)

        # 3. 应用softmax得到权重, weights 的 shape 为 [B, 2], 且每行之和为1
        weights = F.softmax(mask_scores, dim=1)

        # 4. 提取第一列作为mv_p1的权重w
        w = weights[:, 0].unsqueeze(-1) # shape -> [B, 1]

        # --- 后续逻辑不变 ---
        w = w.unsqueeze(-1).unsqueeze(-1) # 扩展维度以进行广播乘法, shape -> [B, 1, 1, 1]
        mv_synth = w * mv_p1 + (1 - w) * mv_p2
        
        # --- 使用“最大绝对值选择法”进行残差汇聚 (未改变) ---
        res_agg = torch.where(torch.abs(res_p1) >= torch.abs(res_p2), res_p1, res_p2)
        
        mask_synth = torch.max(mask_p1, mask_p2)

        # --- 调用 MMFT_Encoder (未改变) ---
        all_motion_inputs = self.motion_encoder(i_frames_raw, mv_synth, res_agg, mask_synth)
        
        motion_summary = all_motion_inputs.view(batch_size, self.num_frames, self.feature_dim)
        v_features_teacher = self.temporal_fusion(visual_output, motion_summary, video_mask)
        
        return v_features_teacher