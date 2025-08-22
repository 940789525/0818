# file: modules/teacher_modules.py (最终优化版 - “精简融合”架构)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- RoPE的核心实现 (未修改) ---

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        self.register_buffer("cos_cached", freqs.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", freqs.sin()[None, None, :, :], persistent=False)

    def forward(self, seq_len):
        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...],
        )

def apply_rotary_pos_emb(q, k, cos, sin):
    q_half1, q_half2 = q.chunk(2, dim=-1)
    k_half1, k_half2 = k.chunk(2, dim=-1)
    q_rotated = torch.cat(
        (q_half1 * cos - q_half2 * sin, q_half1 * sin + q_half2 * cos), dim=-1
    )
    k_rotated = torch.cat(
        (k_half1 * cos - k_half2 * sin, k_half1 * sin + k_half2 * cos), dim=-1
    )
    return q_rotated, k_rotated

class RoPEAttention(nn.Module):
    def __init__(self, embed_dim, nhead):
        super().__init__()
        self.nhead = nhead
        self.head_dim = embed_dim // nhead
        assert self.head_dim * nhead == embed_dim, "embed_dim must be divisible by nhead"
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.rotary_emb = RotaryPositionalEmbedding(dim=self.head_dim)

    def forward(self, query, key, value, key_padding_mask=None):
        B, S_q, _ = query.shape; B, S_kv, _ = key.shape
        q = self.q_proj(query).view(B, S_q, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, S_kv, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, S_kv, self.nhead, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rotary_emb(S_q)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = (attn_probs @ v).transpose(1, 2).reshape(B, S_q, -1)
        return self.out_proj(attn_output)

# --- 空间编码器所需的模块 ---

class SwiGLU_FFN(nn.Module):
    def __init__(self, embed_dim, hidden_dim_multiplier=4):
        super().__init__()
        hidden_dim = int(2 / 3 * hidden_dim_multiplier * embed_dim)
        self.w1 = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, embed_dim, bias=False)
    def forward(self, x):
        gate = F.silu(self.w1(x)); content = self.w3(x)
        return self.w2(gate * content)

# --- 【新改动】: 最终版 - 极致精简的高效融合模块 ---
class LeanFusionBlock(nn.Module):
    def __init__(self, embed_dim, nhead):
        super().__init__()
        self.norm_i1 = nn.RMSNorm(embed_dim)
        self.norm_i2 = nn.RMSNorm(embed_dim)
        self.norm_i3 = nn.RMSNorm(embed_dim)

        # 只保留图像的自注意力
        self.self_attn_i = nn.MultiheadAttention(embed_dim, nhead, bias=False, batch_first=True)
        
        # 保留图像查询运动的交叉注意力
        self.cross_attn = nn.MultiheadAttention(embed_dim, nhead, bias=False, batch_first=True)
        
        self.ffn = SwiGLU_FFN(embed_dim)

    def forward(self, i_tokens, m_tokens, r_tokens):
        # 1. 图像进行自注意力
        i_tokens = i_tokens + self.self_attn_i(self.norm_i1(i_tokens), self.norm_i1(i_tokens), self.norm_i1(i_tokens))[0]

        # 2. 图像查询原始的运动和残差信息
        motion_kv = torch.cat([m_tokens, r_tokens], dim=1)
        i_tokens = i_tokens + self.cross_attn(self.norm_i2(i_tokens), motion_kv, motion_kv)[0]
        
        # 3. FFN
        i_tokens = i_tokens + self.ffn(self.norm_i3(i_tokens))
        
        # m_tokens 和 r_tokens 在这个block中不被更新，直接透传
        return i_tokens, m_tokens, r_tokens

# --- 【新改动】: 使用最终版高效融合模块重构 MMFT_Encoder ---
class MMFT_Encoder(nn.Module):
    """
    【版本11 - 精简融合架构】
    回归初心，在原始架构基础上移除冗余的自注意力模块，实现最高效率。
    """
    def __init__(self, output_dim=512, img_size=224, patch_size=32, embed_dim=256, nhead=4, num_layers=2):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        self.patch_embed_i = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.patch_embed_m = nn.Conv2d(2, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.patch_embed_r = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        self.blocks = nn.ModuleList(
            [LeanFusionBlock(embed_dim, nhead) for _ in range(num_layers)]
        )

        self.norm = nn.RMSNorm(embed_dim)
        self.head = nn.Linear(embed_dim, output_dim)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.RMSNorm)):
            nn.init.constant_(m.weight, 1.0)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            
    def forward(self, i_frame_image, mv, res, motion_mask):
        B = i_frame_image.shape[0]

        i_tokens = self.patch_embed_i(i_frame_image).flatten(2).transpose(1, 2)
        m_tokens = self.patch_embed_m(mv).flatten(2).transpose(1, 2)
        r_tokens = self.patch_embed_r(res).flatten(2).transpose(1, 2)
        
        i_tokens = i_tokens + self.pos_embed
        m_tokens = m_tokens + self.pos_embed
        r_tokens = r_tokens + self.pos_embed

        for blk in self.blocks:
            i_tokens, m_tokens, r_tokens = blk(i_tokens, m_tokens, r_tokens)
        
        fused_feature = self.norm(i_tokens.mean(dim=1))
        return self.head(fused_feature)

# --- 时序模块 (未修改) ---
class RoPE_TransformerEncoderBlock(nn.Module):
    def __init__(self, feature_dim, num_heads, kernel_size=3):
        super().__init__()
        self.norm1 = nn.RMSNorm(feature_dim)
        self.attention = RoPEAttention(feature_dim, num_heads)
        self.norm2 = nn.RMSNorm(feature_dim)
        self.ffn = SwiGLU_FFN(feature_dim)

    def forward(self, x, mask=None):
        x = x + self.attention(self.norm1(x), self.norm1(x), self.norm1(x), key_padding_mask=mask)
        x = x + self.ffn(self.norm2(x))
        return x

class TeacherTemporalFusion(nn.Module):
    def __init__(self, feature_dim, max_seq_length=20, num_transformer_layers=2, num_transformer_heads=4):
        super(TeacherTemporalFusion, self).__init__()
        self.cross_attention_fusion = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_transformer_heads, bias=False)
        self.norm_i = nn.RMSNorm(feature_dim)
        self.norm_m = nn.RMSNorm(feature_dim)
        self.transformer_encoder_blocks = nn.ModuleList(
            [RoPE_TransformerEncoderBlock(feature_dim, num_transformer_heads, kernel_size=3) for _ in range(num_transformer_layers)]
        )
    def forward(self, i_features_sequence, motion_summary, video_mask):
        attn_mask = (video_mask == 0)
        enriched_sequence_tensor = i_features_sequence + self.cross_attention_fusion(
            query=self.norm_i(i_features_sequence).permute(1,0,2),
            key=self.norm_m(motion_summary).permute(1,0,2),
            value=self.norm_m(motion_summary).permute(1,0,2),
            key_padding_mask=attn_mask
        )[0].permute(1,0,2)
        sequence_output = enriched_sequence_tensor * video_mask.unsqueeze(-1)
        for block in self.transformer_encoder_blocks:
            sequence_output = block(sequence_output, mask=attn_mask)
        return sequence_output * video_mask.unsqueeze(-1)