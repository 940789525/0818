# file: modules/modeling_teacher_only.py (已修改)

import torch
import torch.nn as nn

# 【改动1】: 导入新的 MMFT_Encoder, TeacherTemporalFusion
# MotionEncoder 和 ResidualFusionModule 不再需要
from .teacher_modules import MMFT_Encoder, TeacherTemporalFusion

class TeacherOnlyModel(nn.Module):
    """
    一个只包含教师分支的模型，用于独立评估教师模型的性能。
    【版本2】: 使用MMFT_Encoder替换原有的MotionEncoder。
    """
    def __init__(self, feature_dim=512, num_frames=12, num_p_per_group=2):
        super(TeacherOnlyModel, self).__init__()
        self.feature_dim = feature_dim
        self.num_frames = num_frames
        self.num_p_per_group = num_p_per_group
        
        # 【改动2】: 实例化新的 MMFT_Encoder
        # 注意: MMFT_Encoder的输出维度由其内部的head层决定，我们设为feature_dim
        self.motion_encoder = MMFT_Encoder(output_dim=self.feature_dim)
        
        self.temporal_fusion = TeacherTemporalFusion(feature_dim=self.feature_dim, max_seq_length=self.num_frames)

    def forward(self, visual_output, i_frames_raw, mv, res, motion_mask, video_mask):
        """
        前向传播逻辑
        """
        
        batch_size = visual_output.shape[0]
        num_groups = batch_size * self.num_frames
        
        # --- P帧数据预融合流程 (与你提供的文件保持一致) ---
        mv_reshaped = mv.view(num_groups, self.num_p_per_group, *mv.shape[1:])
        res_reshaped = res.view(num_groups, self.num_p_per_group, *res.shape[1:])
        
        mv_p1, mv_p2 = mv_reshaped[:, 0], mv_reshaped[:, 1]
        res_p1, res_p2 = res_reshaped[:, 0], res_reshaped[:, 1]

        mv_synth = mv_p1 + mv_p2
        
        mask_reshaped = motion_mask.view(num_groups, self.num_p_per_group, *motion_mask.shape[1:])
        mask_p1, mask_p2 = mask_reshaped[:, 0], mask_reshaped[:, 1]
        mask_synth = torch.max(mask_p1, mask_p2)

        # --- 【改动3】: 调用新的 MMFT_Encoder ---
        # 输入与你之前测试时保持一致，使用 mv_synth 和 res_p1
        all_motion_inputs = self.motion_encoder(i_frames_raw, mv_synth, res_p1, mask_synth)
        
        motion_summary = all_motion_inputs.view(batch_size, self.num_frames, self.feature_dim)
        v_features_teacher = self.temporal_fusion(visual_output, motion_summary, video_mask)
        
        # 返回序列特征，与你文件中的设定保持一致
        return v_features_teacher