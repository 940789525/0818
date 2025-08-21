# file: loss_modules.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------------
# 模块一：独立的损失函数定义
# --------------------------------------------------------------------------------

def margin_infonce_loss(features_a, features_b, temperature=0.07, margin=0.1):
    """
    L1 (升级版): 带裕度的InfoNCE主任务损失。
    """
    # 1. 归一化特征并计算原始的相似度矩阵 (Logits)
    features_a = F.normalize(features_a, p=2, dim=-1)
    features_b = F.normalize(features_b, p=2, dim=-1)
    # 增加epsilon防止除零
    logits = torch.matmul(features_a, features_b.t()) / (temperature + 1e-8) 
    
    batch_size = features_a.shape[0]
    # 提取对角线上的正样本logit
    positive_logits = torch.diag(logits)
    
    # 2. 创建一个对角线为True的布尔掩码
    mask = torch.eye(batch_size, dtype=torch.bool, device=logits.device)
    
    # 3. 复制一份logits矩阵，并使用掩码精确地将对角线上的值替换为 (正样本logit - margin)
    logits_with_margin = logits.clone()
    logits_with_margin[mask] = positive_logits - margin
    
    # 4. 使用修改后的logits计算标准的交叉熵损失
    labels = torch.arange(batch_size, device=features_a.device)
    loss = nn.CrossEntropyLoss()(logits_with_margin, labels)
    return loss

def ranking_distillation_loss(logits_student, logits_teacher, margin=0.1):
    """ L3 (升级版): 基于排序关系的关系知识蒸馏损失。 """
    logits_teacher = logits_teacher.detach()
    batch_size = logits_student.size(0)
    if batch_size <= 1:
        return torch.tensor(0.0, device=logits_student.device)

    s_expanded1 = logits_student.unsqueeze(2).expand(-1, -1, batch_size)
    s_expanded2 = logits_student.unsqueeze(1).expand(-1, batch_size, -1)
    t_expanded1 = logits_teacher.unsqueeze(2).expand(-1, -1, batch_size)
    t_expanded2 = logits_teacher.unsqueeze(1).expand(-1, batch_size, -1)
    diff_s = s_expanded1 - s_expanded2
    diff_t = t_expanded1 - t_expanded2
    target = torch.sign(diff_t)
    # 1. 计算每个元素的损失，但先不求平均
    # reduction='none' 会返回一个和输入形状相同的损失张量
    loss_tensor = F.margin_ranking_loss(diff_s, torch.zeros_like(diff_s), target, margin=margin, reduction='none')

    # 2. 创建一个掩码，用于忽略对角线元素 (j == k)
    batch_size = logits_student.size(0)
    # 创建一个对角线为1，其余为0的矩阵
    identity_matrix = torch.eye(batch_size, device=logits_student.device)
    # 我们需要的是非对角线元素，所以用1减去它，并扩展到正确的形状
    # mask中，非对角线位置为1，对角线位置为0
    mask = 1.0 - identity_matrix.unsqueeze(0).expand_as(loss_tensor)

    # 3. 将损失张量与掩码相乘，对角线损失变为0
    masked_loss = loss_tensor * mask

    # 4. 计算真实平均值：只对非对角线元素求平均
    # 总损失除以非零元素的数量
    final_loss = masked_loss.sum() / mask.sum()

    return final_loss
class DCDLoss(nn.Module):
    """ L2 (升级版): 判别性与一致性蒸馏 (DCD) """
    def __init__(self, temperature=0.1, consistency_weight=1.0):
        super(DCDLoss, self).__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.consistency_weight = consistency_weight
    def forward(self, features_student, features_teacher):
        features_student = F.normalize(features_student, p=2, dim=1)
        features_teacher = F.normalize(features_teacher, p=2, dim=1)
        sim_s_t = torch.matmul(features_student, features_teacher.t()) / (self.temperature + 1e-8)
        labels = torch.arange(features_student.size(0), device=features_student.device)
        loss_discriminative = F.cross_entropy(sim_s_t, labels)
        with torch.no_grad():
            sim_t_t = torch.matmul(features_teacher, features_teacher.t()) / (self.temperature + 1e-8)
        loss_consistent = F.mse_loss(sim_s_t, sim_t_t)
        total_loss = loss_discriminative + self.consistency_weight * loss_consistent
        return total_loss

# --------------------------------------------------------------------------------
# 模块二：封装所有损失计算和调度逻辑的主类
# --------------------------------------------------------------------------------

class DistillationLoss(nn.Module):
    """
    【最终版】：L1使用裕度对比损失, L2使用DCD, L3使用排序蒸馏。
    """
    def __init__(self, final_alpha=1.0, final_beta=0.5, warmup_proportion=0.2, increase_proportion=0.3, l1_margin=0.1):
        super(DistillationLoss, self).__init__()
        self.final_alpha = final_alpha
        self.final_beta = final_beta
        self.warmup_proportion = warmup_proportion
        self.increase_proportion = increase_proportion
        self.l1_margin = l1_margin
        
        if self.increase_proportion <= 0:
            self.increase_proportion = 1.0
        
        self.dcd_loss = DCDLoss(temperature=0.1, consistency_weight=1.0)
        
        # --- 新增代码: 初始化上次打印的进度 ---
        self.last_print_progress = 0.0

    def _get_dynamic_weights(self, training_progress):
        alpha, beta = 0.0, 0.0
        if training_progress >= self.warmup_proportion:
            progress_in_increase_phase = (training_progress - self.warmup_proportion) / self.increase_proportion
            schedule_progress = min(1.0, progress_in_increase_phase)
            alpha = self.final_alpha * schedule_progress
            beta = self.final_beta * schedule_progress
        return alpha, beta

    def _masked_mean(self, features, mask):
        mask_expanded = mask.unsqueeze(-1).expand_as(features)
        return (features * mask_expanded).sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)

    def forward(self, v_features_teacher, v_features_student, t_features, video_mask, training_progress):
        alpha, beta = self._get_dynamic_weights(training_progress)
        v_teacher_summary = self._masked_mean(v_features_teacher, video_mask)
        v_student_summary = self._masked_mean(v_features_student, video_mask)

        loss_main_teacher = margin_infonce_loss(v_teacher_summary, t_features, margin=self.l1_margin)
        loss_main_student = margin_infonce_loss(v_student_summary, t_features, margin=self.l1_margin)
        
        loss_feat = self.dcd_loss(v_student_summary, v_teacher_summary)
        
        logits_teacher = v_teacher_summary @ t_features.t()
        logits_student = v_student_summary @ t_features.t()
        loss_logits = ranking_distillation_loss(logits_student, logits_teacher)

        if training_progress < self.warmup_proportion:
            loss_main = loss_main_teacher
            loss_total = loss_main
        else:
            loss_main = (loss_main_teacher + loss_main_student) * 0.5
            loss_total = loss_main + alpha * loss_feat + beta * loss_logits
            
        # --- 新增代码: 定期打印各个损失分量的值 ---
        if training_progress >= self.last_print_progress + 0.05:
            print("\n" + "="*20 + f" DEBUG LOSSES (Progress: {training_progress:.2%}) " + "="*20)
            print(f"  - loss_main:   {loss_main.item():.6f}")
            print(f"  - loss_feat:   {loss_feat.item():.6f} (alpha: {alpha:.4f})")
            print(f"  - loss_logits: {loss_logits.item():.6f} (beta: {beta:.4f})")
            print("="*68 + "\n")
            # 更新记录，防止在同一个进度区间内重复打印
            self.last_print_progress = training_progress

        return {
            "loss_total": loss_total,
            "loss_main": loss_main,
            "loss_feat": loss_feat,
            "loss_logits": loss_logits,
            "alpha": alpha,
            "beta": beta
        }