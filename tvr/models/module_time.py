# TimeRouter.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeRouter(nn.Module):
    """
    轻量“时间路由器”：
    - 输入（二选一，自动适配）：
        x: [B*T, 1+N, W]  或  [B, T, 1+N, W]   （已做 patch-embed、加pos、ln_pre）
        video_mask: [B, T]（1=有效帧, 0=padding）
    - 输出：
        x_reduced: [B, K, 1+N_prime, W]    （仅保留 K 帧，每帧仅保留 N' 个 patch）
        new_mask:  [B, K]                  （保留帧的mask，当前全部=1）
        meta: dict                         （K/T、分数熵、选中索引、每帧patch预算等）

    设计要点（极轻 + 高召回）：
      * 帧级代理：patch均值  + 相邻差分  + 新颖度（与历史均值的夹角），复杂度 O(B·T·W)
      * Top-K + 时间NMS + 熵自适应：分数不确定时自动多保留帧
      * Patch 分档预算：{1.0, 0.5, 0.25}，低维相似度选 Top-m 的 patch，复杂度 O(B·T·N·w')
      * 批内长度对齐到 K_global 与 N'_global，便于继续走标准 Transformer
    """

    def __init__(
        self,
        width: int,                 # W（与视觉塔宽度一致）
        max_frames: int,            # T_max（用于安全边界/日志）
        proxy_dim: int = 32,        # w'，低维相似度空间
        topk_ratio: float = 0.5,    # K ≈ ceil(ratio * T_valid)
        min_keep: int = 4,          # 每个视频至少保留的帧数
        temporal_nms: int = 0,      # 时间NMS半径（抑制相邻冗余），单位=帧
        patch_keep_factors: Tuple[float, ...] = (1.0, 0.5, 0.25)  # 每帧patch预算分档比例
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.max_frames = int(max_frames)
        self.proxy_dim = int(proxy_dim)
        self.topk_ratio = float(topk_ratio)
        self.min_keep = int(min_keep)
        self.temporal_nms = max(0, int(temporal_nms))
        self.patch_keep_factors = tuple(sorted(patch_keep_factors, reverse=True))  # e.g. (1.0, 0.5, 0.25)

        # 低维投影（帧/patch都投到 w' 维，做极轻相似度）
        self.frame_proj = nn.Linear(self.width, self.proxy_dim, bias=False)
        self.patch_proj = nn.Linear(self.width, self.proxy_dim, bias=False)

        # 帧分数（内容强度 + 运动 + 新颖度）→ 标量logit
        self.score_mlp = nn.Sequential(
            nn.LayerNorm(self.proxy_dim * 2 + 1),
            nn.Linear(self.proxy_dim * 2 + 1, self.proxy_dim),
            nn.GELU(),
            nn.Linear(self.proxy_dim, 1),
        )

        # 每帧patch预算（分档）所需的标量概率
        self.budget_head = nn.Sequential(
            nn.LayerNorm(self.proxy_dim),
            nn.Linear(self.proxy_dim, 1),
        )

    # ---------- helpers ----------

    @staticmethod
    def _entropy_softmax(logits: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
        p = torch.softmax(logits, dim=dim)
        return -(p * p.clamp_min(eps).log()).sum(dim=dim)

    def _temporal_nms_topk(self, scores_b: torch.Tensor, valid_mask_b: torch.Tensor, k: int) -> torch.Tensor:
        """贪心时间NMS：按分数降序选帧，抑制 ±r 邻域，直到取满k或没有可选。
        scores_b: [T], valid_mask_b: [T] in {0,1}  →  return: idx (升序保持时序)
        """
        T = scores_b.numel()
        sc = scores_b.masked_fill(~valid_mask_b.bool(), float("-inf"))
        order = torch.argsort(sc, descending=True)
        taken = torch.zeros(T, dtype=torch.bool, device=scores_b.device)
        picked = []
        for t in order.tolist():
            if len(picked) >= k or not torch.isfinite(sc[t]):
                break
            if taken[t]:
                continue
            picked.append(t)
            if self.temporal_nms > 0:
                l = max(0, t - self.temporal_nms)
                r = min(T, t + self.temporal_nms + 1)
                taken[l:r] = True
        if not picked:  # 极端回退
            picked = [0]
        return torch.tensor(sorted(picked), device=scores_b.device, dtype=torch.long)

    # ---------- forward ----------

    def forward(
        self,
        x: torch.Tensor,           # [B*T, 1+N, W]  或  [B, T, 1+N, W]
        video_mask: torch.Tensor,  # [B, T] in {0,1}
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        assert x.dim() in (3, 4), f"expect x dims 3/4, got {x.shape}"
        assert video_mask.dim() == 2, f"expect video_mask [B,T], got {video_mask.shape}"

        device = x.device
        B, T = video_mask.shape

        # 统一成 [B, T, 1+N, W]
        if x.dim() == 3:  # [B*T, 1+N, W] → [B, T, 1+N, W]
            BT, L, W = x.shape
            assert BT == B * T, f"shape mismatch: x={x.shape}, mask={video_mask.shape}"
            x_b_t_l_w = x.view(B, T, L, W)
        else:
            x_b_t_l_w = x
            _, _, L, W = x_b_t_l_w.shape

        assert L >= 2, "expect 1+N tokens per frame"
        N = L - 1

        # === 1) 帧级代理：内容强度 + 运动 + 新颖度（超轻） ===
        patch = x_b_t_l_w[:, :, 1:, :]        # [B,T,N,W]
        p_t = patch.mean(dim=2)               # [B,T,W]  全局摘要
        p_low = self.frame_proj(p_t)          # [B,T,w']

        # 运动：相邻差分（首帧归零）
        p_prev = torch.roll(p_low, 1, dims=1)
        p_prev[:, 0, :] = 0.0
        delta_low = p_low - p_prev            # [B,T,w']
        # delta_low[:, 0, :] = 0.0

        # === 双向新颖度（Bi-directional Novelty）：与“过去窗口均值”和“未来窗口均值”各算一次 1-cos，取更大者 ===
        w_hist = 2  # 过去/未来窗口长度，可与原设定一致

        hist_mean_past = []
        hist_mean_fut  = []
        for t in range(T):
            # ----- 过去窗口 [t-w_hist, t) -----
            l = max(0, t - w_hist)
            if t > l:
                seg = p_low[:, l:t, :]                               # [B, Lp, w']
                seg_mask = video_mask[:, l:t].unsqueeze(-1).bool()   # [B, Lp, 1]
                seg = seg.masked_fill(~seg_mask, 0.0)
                denom = seg_mask.sum(dim=1, keepdim=True).clamp_min(1)
                hist_mean_past.append(seg.sum(dim=1, keepdim=True) / denom)  # [B,1,w']
            else:
                hist_mean_past.append(p_low[:, t:t+1, :])            # 没有过去帧就退回自身

            # ----- 未来窗口 (t, t+w_hist] -----
            r = min(T, t + 1 + w_hist)
            if r > t + 1:
                segf = p_low[:, t+1:r, :]                             # [B, Lf, w']
                segf_mask = video_mask[:, t+1:r].unsqueeze(-1).bool() # [B, Lf, 1]
                segf = segf.masked_fill(~segf_mask, 0.0)
                denomf = segf_mask.sum(dim=1, keepdim=True).clamp_min(1)
                hist_mean_fut.append(segf.sum(dim=1, keepdim=True) / denomf) # [B,1,w']
            else:
                hist_mean_fut.append(p_low[:, t:t+1, :])              # 没有未来帧就退回自身

        hist_mean_past = torch.cat(hist_mean_past, dim=1)  # [B,T,w']
        hist_mean_fut  = torch.cat(hist_mean_fut,  dim=1)  # [B,T,w']

        # 计算两个方向的新颖度，并取最大值（更“新”的那一侧）
        p_n = F.normalize(p_low,         dim=-1, eps=1e-6)
        h_p = F.normalize(hist_mean_past,dim=-1, eps=1e-6)
        h_f = F.normalize(hist_mean_fut, dim=-1, eps=1e-6)

        novelty_past = 1.0 - (p_n * h_p).sum(dim=-1, keepdim=True)  # [B,T,1]
        novelty_fut  = 1.0 - (p_n * h_f).sum(dim=-1, keepdim=True)  # [B,T,1]
        novelty = torch.maximum(novelty_past, novelty_fut)           # [B,T,1]

        # 帧分数 logits（极小 MLP）
        # === 轻量、可解释的帧级打分：s = α·z(novelty) + β·z(||delta_low||) ===
        # novelty: [B,T,1] → [B,T]，表示“方向新颖度”；delta_low: [B,T,w′]，表示“相邻变化幅度”
        nov = novelty.squeeze(-1)                 # [B,T]
        # --- 新 Δ（方向性相邻差）---
        pl  = F.normalize(p_low, dim=-1, eps=1e-6)            # [B,T,w']
        plp = torch.roll(pl, 1, dims=1)
        plp[:, 0, :] = pl[:, 0, :]                            # 让 t=0 与自身对齐 → Δ_dir[0]=0
        delta_dir = 1.0 - (pl * plp).sum(dim=-1)              # [B,T] ∈ [0,2]

        # 逐视频 z-score 并线性打分（保持 α/β 不变即可，先不动其它）
        def z(a): 
            return (a - a.mean(dim=1, keepdim=True)) / (a.std(dim=1, keepdim=True) + 1e-6)

        nov = novelty.squeeze(-1)          # [B,T]
        n_hat = z(nov)
        d_hat = z(delta_dir)               # 用方向性 Δ

        alpha, beta = 0.6, 0.4
        frame_logits = (alpha * n_hat + beta * d_hat)
        frame_logits = frame_logits.masked_fill(~video_mask.bool(), float("-inf"))

        # 不确定性熵（越大说明分数很平）
        ent = self._entropy_softmax(frame_logits, dim=1)       # [B]

        # 动态 K（按有效 T 与不确定性）
        valid_T = video_mask.sum(dim=1)                        # [B]
        k_target = torch.ceil(self.topk_ratio * valid_T.float()).to(torch.long)
        k_target = torch.clamp(k_target, min=self.min_keep, max=T)
        k_target = torch.clamp(k_target + (ent > (math.log(max(T, 2)) * 0.85)).long(), max=T)

        # 逐视频 Top-K + 时间NMS
        sel_indices = []
        k_list = []
        for b in range(B):
            idx_b = self._temporal_nms_topk(frame_logits[b], video_mask[b], int(k_target[b].item()))
            sel_indices.append(idx_b)
            k_list.append(idx_b.numel())
        K_global = int(max(k_list)) if k_list else int(k_target.max().item())

        # 对齐到 K_global（不足用次优帧补齐；保持升序）
        idx_full = []
        for b in range(B):
            idx_b = sel_indices[b]
            if idx_b.numel() < K_global:
                sc = frame_logits[b]
                order = torch.argsort(sc, descending=True)
                used = set(idx_b.tolist())
                extra = []
                for j in order.tolist():
                    if len(extra) + idx_b.numel() >= K_global:
                        break
                    if j in used or not torch.isfinite(sc[j]):
                        continue
                    extra.append(j)
                    used.add(j)
                if extra:
                    idx_b = torch.tensor(sorted(idx_b.tolist() + extra), device=device, dtype=torch.long)
            else:
                idx_b = idx_b[:K_global]
            idx_full.append(idx_b)
        idx_full = torch.stack(idx_full, dim=0)                # [B,K]
        new_mask = torch.ones(B, K_global, dtype=torch.bool, device=device)

        # Gather 选帧 → [B,K,1+N,W]
        b_idx = torch.arange(B, device=device).unsqueeze(-1).expand(B, K_global)
        x_sel = x_b_t_l_w[b_idx, idx_full]

        meta = {
            "K": torch.tensor(K_global, device=device),
            "K_over_T": (k_target.float() / valid_T.clamp_min(1.0)),  # [B]
            "frame_scores": frame_logits,                              # [B,T]
            "frame_entropy": ent,                                      # [B]
            "selected_idx": idx_full,                                  # [B,K]
            # 下面两项方便你继续诊断/画图（可留可去）
            "novelty": novelty.squeeze(-1),                            # [B,T]
            "delta_dir": delta_dir,                                    # [B,T]
        }


        use_patch_pruning = False
        if not use_patch_pruning:
            x_reduced = x_sel                      # [B,K,1+N,W] —— 不再帧内裁剪
            new_mask = torch.ones(B, K_global, dtype=torch.bool, device=device)
        else:
            # === 2) Patch 预算分档 + 低维相似度选 Top-m ===
            p_low_sel = p_low[b_idx, idx_full]                     # [B,K,w']
            budget_logit = self.budget_head(p_low_sel).squeeze(-1) # [B,K]
            budget_prob = torch.sigmoid(budget_logit)              # [B,K] ∈ [0,1]

            # 根据桶数生成阈值（均匀切分 [0,1]）
            n_bins = len(self.patch_keep_factors)
            thresholds = self._even_thresholds(n_bins, device=device)          # shape [n_bins-1]
            # bucketize：大概率 → 大bin id
            bins = torch.bucketize(budget_prob, thresholds)
            # 因子按“升序”映射到bins（低概率 → 小因子，高概率 → 大因子）
            keep_factors = torch.tensor(sorted(self.patch_keep_factors), device=device)
            keep_ratio = keep_factors[bins]                                     # [B,K]
            m_t = torch.clamp((keep_ratio * N).ceil().to(torch.long), min=1, max=N)  # [B,K]

            # 低维相似度（帧方向 vs patch）
            patch_sel = x_sel[:, :, 1:, :]                    # [B,K,N,W]
            patch_low = self.patch_proj(patch_sel)            # [B,K,N,w']
            p_dir = F.normalize(p_low_sel.unsqueeze(2), dim=-1, eps=1e-6)  # [B,K,1,w']
            t_dir = F.normalize(patch_low, dim=-1, eps=1e-6)               # [B,K,N,w']
            sim = (p_dir * t_dir).sum(dim=-1)                              # [B,K,N]

            # --- N' 采用“批内分位数”，而非最大值（让“多数帧的预算”真正生效）
            flat_m = m_t.flatten().float()
            qv = float(self._quantile_safe(flat_m, self.nprime_quantile).item())
            N_prime = int(max(self.nprime_min, min(math.ceil(qv), N)))

            # 取每帧前 N' 个候选；对超配额 patch 置零（保持长度统一但不贡献信息）
            _, idx_patch = torch.topk(sim, k=N_prime, dim=2, largest=True, sorted=True)  # [B,K,N']
            b_idx2 = torch.arange(B, device=device).view(B, 1, 1).expand(B, idx_patch.size(1), idx_patch.size(2))
            k_idx2 = torch.arange(idx_patch.size(1), device=device).view(1, idx_patch.size(1), 1).expand_as(idx_patch)
            patch_top = patch_sel[b_idx2, k_idx2, idx_patch]   # [B,K,N',W]

            # 超过该帧预算的部分置零
            idx_range = torch.arange(N_prime, device=device).view(1, 1, N_prime)      # [1,1,N']
            patch_valid = (idx_range < m_t.unsqueeze(-1))                               # [B,K,N'] bool
            patch_top = patch_top * patch_valid.unsqueeze(-1)                           # [B,K,N',W]

            # 拼回 CLS
            cls_tok = x_sel[:, :, :1, :]                       # [B,K,1,W]
            x_reduced = torch.cat([cls_tok, patch_top], dim=2) # [B,K,1+N',W]

            meta: Dict[str, torch.Tensor] = {
                "K": torch.tensor(K_global, device=device),
                "N_prime": torch.tensor(N_prime, device=device),
                "K_over_T": (k_target.float() / valid_T.clamp_min(1.0)),  # [B]
                "frame_scores": frame_logits,                              # [B,T]
                "frame_entropy": ent,                                      # [B]
                "selected_idx": idx_full,                                  # [B,K]
                "patch_budget_m": m_t,                                     # [B,K]
            }
        return x_reduced, new_mask, meta
