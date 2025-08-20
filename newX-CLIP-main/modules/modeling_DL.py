# file: modules/modeling_DL.py

import torch
import torch.nn as nn

# 从teacher_modules中导入更新后的模块
# Python的导入机制会自动加载我们最新的、包含MVCM的teacher_modules.py
from .teacher_modules import  TeacherTemporalFusion, ResidualFusionModule
from .student_modules import DeltaPredictor

class DistillationHead(nn.Module):
    """
    师生蒸馏头部模块。
    【更新版】：集成了使用MVCM的MotionEncoder，可以接收原始I帧作为输入。
    """
    def __init__(self, feature_dim=512, num_frames=12, num_p_per_group=2):
        super(DistillationHead, self).__init__()
        self.feature_dim = feature_dim
        self.num_frames = num_frames
        self.num_p_per_group = num_p_per_group
        
        # 1. 教师分支的模块
        #    这里的MotionEncoder现在是包含了MVCM的重构版本
        self.motion_encoder = MotionEncoder(feature_dim=self.feature_dim)
        self.residual_fuser = ResidualFusionModule()
        self.temporal_fusion = TeacherTemporalFusion(feature_dim=self.feature_dim, max_seq_length=self.num_frames)
        
        # 2. 学生分支的模块
        self.delta_predictor = DeltaPredictor(feature_dim=self.feature_dim)

    def forward(self, visual_output, i_frames_raw, mv, res, motion_mask, video_mask):
        """
        前向传播函数。
        【更新版】：增加了i_frames_raw参数。
        
        :param visual_output: I帧序列特征 [N, Seq, C]
        :param i_frames_raw: 原始I帧RGB张量 [N*Seq, 3, H, W]
        :param mv: 运动矢量 [N*Seq*k, 2, H, W]
        :param res: 残差 [N*Seq*k, 3, H, W]
        :param motion_mask: 运动掩码 [N*Seq*k, 1, H, W]
        :param video_mask: 有效I帧掩码 [N, Seq]
        """
        batch_size = visual_output.shape[0]
        # --- P帧数据预融合流程 ---
        num_groups = batch_size * self.num_frames
        
        mv_reshaped = mv.view(num_groups, self.num_p_per_group, *mv.shape[1:])
        res_reshaped = res.view(num_groups, self.num_p_per_group, *res.shape[1:])
        mask_reshaped = motion_mask.view(num_groups, self.num_p_per_group, *motion_mask.shape[1:])

        mv_p1, mv_p2 = mv_reshaped[:, 0], mv_reshaped[:, 1]
        res_p1, res_p2 = res_reshaped[:, 0], res_reshaped[:, 1]
        mask_p1, mask_p2 = mask_reshaped[:, 0], mask_reshaped[:, 1]

        mv_synth = mv_p1 + mv_p2
        res_delta = self.residual_fuser(res_p1, res_p2, mv_p1, mv_p2)
        res_synth = res_p1 + res_delta
        mask_synth = torch.max(mask_p1, mask_p2)

        # --- 教师分支流程 ---
        
        # --- 【核心修改点】 ---
        # 调用motion_encoder时，传入原始I帧 (i_frames_raw) 作为其第一个参数
        all_motion_inputs = self.motion_encoder(i_frames_raw, mv_synth, res_synth, mask_synth)
        
        # --- 后续流程完全不变 ---
        motion_summary = all_motion_inputs.view(batch_size, self.num_frames, self.feature_dim)
        v_features_teacher = self.temporal_fusion(visual_output, motion_summary, video_mask)
        
        # --- 学生分支流程 (完全不变) ---
        masked_visual_output = visual_output * video_mask.unsqueeze(-1)
        delta_student = self.delta_predictor(masked_visual_output)
        v_features_student = (masked_visual_output + delta_student) * video_mask.unsqueeze(-1)

        return v_features_teacher, v_features_student