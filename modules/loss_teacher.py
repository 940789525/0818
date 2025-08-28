# file: loss_modules.py (专为独立训练教师模型设计)

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------------
# 核心损失函数及其辅助模块
# --------------------------------------------------------------------------------

def symmetric_infonce_loss(features_a, features_b, temperature=0.07):
    """
    辅助函数：CLIP中使用的标准对称InfoNCE损失。
    用于计算粗粒度（全局）特征之间的对比损失。
    """
    # 归一化特征
    features_a = F.normalize(features_a, p=2, dim=-1)
    features_b = F.normalize(features_b, p=2, dim=-1)

    # 计算logits
    logits_a_to_b = torch.matmul(features_a, features_b.t()) / (temperature + 1e-8)
    logits_b_to_a = logits_a_to_b.t()

    batch_size = features_a.shape[0]
    labels = torch.arange(batch_size, device=features_a.device)

    # 计算两个方向的交叉熵损失
    loss_a = nn.CrossEntropyLoss()(logits_a_to_b, labels)
    loss_b = nn.CrossEntropyLoss()(logits_b_to_a, labels)

    # 取平均
    return (loss_a + loss_b) / 2

def _calculate_fine_grained_similarity(v_fine, t_fine, video_mask, text_mask):
    """
    辅助函数：计算所有视频-文本配对之间的细粒度相似度得分。
    返回一个 [B, B] 的矩阵 S_fine，其中 S_fine[i, j] 代表 video_i 和 text_j 之间的匹配分数。
    """
    B, num_frames, D = v_fine.shape
    _, num_tokens, _ = t_fine.shape

    # 归一化细粒度特征
    v_fine = F.normalize(v_fine, p=2, dim=-1)
    t_fine = F.normalize(t_fine, p=2, dim=-1)

    # 扩展维度以便计算所有配对
    v_fine_exp = v_fine.unsqueeze(1)    # -> [B, 1, num_frames, D]
    t_fine_exp = t_fine.unsqueeze(0)    # -> [1, B, num_tokens, D]

    # 计算所有帧和所有词元之间的相似度
    all_sims = torch.matmul(v_fine_exp, t_fine_exp.transpose(2, 3)) # -> [B, B, num_frames, num_tokens]

    # Video-to-Text 方向：对每个video的每一帧，找到最匹配的text词元
    v_mask_exp = video_mask.unsqueeze(1).expand(-1, B, -1)
    v2t_sims, _ = all_sims.max(dim=3) # -> [B, B, num_frames]
    v2t_sims = (v2t_sims * v_mask_exp).sum(dim=2) / (v_mask_exp.sum(dim=2) + 1e-8) # -> [B, B]

    # Text-to-Video 方向：对每个text的每个词元，找到最匹配的video帧
    t_mask_exp = text_mask.unsqueeze(0).expand(B, -1, -1)
    t2v_sims, _ = all_sims.max(dim=2) # -> [B, B, num_tokens]
    t2v_sims = (t2v_sims * t_mask_exp).sum(dim=2) / (t_mask_exp.sum(dim=2) + 1e-8) # -> [B, B]
    
    # 返回对称的细粒度相似度
    return (v2t_sims + t2v_sims) / 2


def margin_free_soft_focusing_loss(
    v_features_fine, t_features_fine,
    v_features_coarse, t_features_coarse,
    video_mask, text_mask,
    coarse_temperature=0.07,
    fine_temperature=0.07,
    lambda_focus=1.0
):
    """
    创新的、无裕度的、更稳健的“软性聚焦”层级对比损失函数。
    这是模型进行视频-文本检索任务的唯一主损失。
    """
    batch_size = v_features_coarse.shape[0]
    device = v_features_coarse.device

    # --- 1. 基础对齐：计算粗粒度损失和相似度矩阵 ---
    loss_coarse = symmetric_infonce_loss(v_features_coarse, t_features_coarse, coarse_temperature)
    
    with torch.no_grad():
        S_coarse = torch.matmul(
            F.normalize(v_features_coarse, p=2, dim=-1),
            F.normalize(t_features_coarse, p=2, dim=-1).t()
        )
    
    # --- 2. 细粒度相似度计算 ---
    S_fine = _calculate_fine_grained_similarity(v_features_fine, t_features_fine, video_mask, text_mask)
    positive_fine_sim = torch.diag(S_fine)

    # --- 3. "软性聚焦"的细粒度损失 (无裕度版本) ---
    
    # a) Video 作为锚点
    S_coarse_v2t = S_coarse.clone()
    S_coarse_v2t.fill_diagonal_(-torch.inf)
    hardness_weights_v2t = F.softmax(S_coarse_v2t / coarse_temperature, dim=1)
    
    S_fine_neg_v2t = S_fine.clone()
    S_fine_neg_v2t.fill_diagonal_(0)
    weighted_avg_neg_sim_v2t = (hardness_weights_v2t * S_fine_neg_v2t).sum(dim=1)
    
    logits_v2t = torch.stack([positive_fine_sim, weighted_avg_neg_sim_v2t], dim=1)
    labels = torch.zeros(batch_size, dtype=torch.long, device=device)
    loss_focus_v2t = F.cross_entropy(logits_v2t / fine_temperature, labels)

    # b) Text 作为锚点，对称计算
    S_coarse_t2v = S_coarse.t().clone()
    S_coarse_t2v.fill_diagonal_(-torch.inf)
    hardness_weights_t2v = F.softmax(S_coarse_t2v / coarse_temperature, dim=1)

    S_fine_neg_t2v = S_fine.t().clone()
    S_fine_neg_t2v.fill_diagonal_(0)
    weighted_avg_neg_sim_t2v = (hardness_weights_t2v * S_fine_neg_t2v).sum(dim=1)

    logits_t2v = torch.stack([positive_fine_sim, weighted_avg_neg_sim_t2v], dim=1)
    loss_focus_t2v = F.cross_entropy(logits_t2v / fine_temperature, labels)

    loss_fine_focused = (loss_focus_v2t + loss_focus_t2v) / 2

    # --- 4. 组合总损失 ---
    total_loss = loss_coarse + lambda_focus * loss_fine_focused
    return total_loss