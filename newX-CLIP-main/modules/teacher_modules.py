# file: modules/teacher_modules.py (最终修正版 - 增加卷积模块)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- RoPE的核心实现 (仅供TeacherTemporalFusion使用) ---

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
        
        # RoPE只对Q和K应用
        cos, sin = self.rotary_emb(S_q) # 注意：假设Q和K有相同的序列长度进行旋转
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = (attn_probs @ v).transpose(1, 2).reshape(B, S_q, -1)
        return self.out_proj(attn_output)

# --- 空间编码器所需的非RoPE模块 (恢复) ---
class CustomCrossAttention(nn.Module):
    def __init__(self, embed_dim, nhead):
        super().__init__()
        self.nhead = nhead
        self.head_dim = embed_dim // nhead
        assert self.head_dim * nhead == embed_dim, "embed_dim must be divisible by nhead"
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, query, key, value, attention_bias=None):
        B, S_q, _ = query.shape; B, S_kv, _ = key.shape
        q = self.q_proj(query).view(B, S_q, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, S_kv, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, S_kv, self.nhead, self.head_dim).transpose(1, 2)
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_bias is not None:
            attn_scores = attn_scores + attention_bias.unsqueeze(1).unsqueeze(2)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = (attn_probs @ v).transpose(1, 2).reshape(B, S_q, -1)
        return self.out_proj(attn_output)

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

class MMFT_Block(nn.Module):
    def __init__(self, embed_dim, nhead):
        super().__init__()
        self.self_attn_i = nn.MultiheadAttention(embed_dim, nhead, bias=False)
        self.self_attn_m = nn.MultiheadAttention(embed_dim, nhead, bias=False)
        self.self_attn_r = nn.MultiheadAttention(embed_dim, nhead, bias=False)
        self.cross_attn = CustomCrossAttention(embed_dim, nhead)
        self.norm1_i = nn.RMSNorm(embed_dim); self.norm1_m = nn.RMSNorm(embed_dim); self.norm1_r = nn.RMSNorm(embed_dim)
        self.norm2_i = nn.RMSNorm(embed_dim); self.norm3_i = nn.RMSNorm(embed_dim)
        self.ffn = SwiGLU_FFN(embed_dim)
    def forward(self, i_tokens, m_tokens, r_tokens, attention_bias=None):
        i_qkv = self.norm1_i(i_tokens).permute(1, 0, 2); i_attn_out, _ = self.self_attn_i(i_qkv, i_qkv, i_qkv); i_tokens = i_tokens + i_attn_out.permute(1, 0, 2)
        m_qkv = self.norm1_m(m_tokens).permute(1, 0, 2); m_attn_out, _ = self.self_attn_m(m_qkv, m_qkv, m_qkv); m_tokens = m_tokens + m_attn_out.permute(1, 0, 2)
        r_qkv = self.norm1_r(r_tokens).permute(1, 0, 2); r_attn_out, _ = self.self_attn_r(r_qkv, r_qkv, r_qkv); r_tokens = r_tokens + r_attn_out.permute(1, 0, 2)
        query = self.norm2_i(i_tokens); kv_tokens = torch.cat([m_tokens, r_tokens], dim=1)
        cross_attn_out = self.cross_attn(query, kv_tokens, kv_tokens, attention_bias=attention_bias)
        i_tokens = i_tokens + cross_attn_out
        i_tokens = i_tokens + self.ffn(self.norm3_i(i_tokens))
        return i_tokens, m_tokens, r_tokens

class MMFT_Encoder(nn.Module):
    def __init__(self, output_dim=512, img_size=224, patch_size=32, embed_dim=256, nhead=4, num_layers=2):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed_i = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.patch_embed_m = nn.Conv2d(2, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.patch_embed_r = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([MMFT_Block(embed_dim, nhead) for _ in range(num_layers)])
        self.norm = nn.RMSNorm(embed_dim)
        self.head = nn.Linear(embed_dim, output_dim)
    def forward(self, i_frame_image, mv, res, motion_mask):
        i_tokens = self.patch_embed_i(i_frame_image).flatten(2).transpose(1, 2); m_tokens = self.patch_embed_m(mv).flatten(2).transpose(1, 2); r_tokens = self.patch_embed_r(res).flatten(2).transpose(1, 2)
        i_tokens = i_tokens + self.pos_embed; m_tokens = m_tokens + self.pos_embed; r_tokens = r_tokens + self.pos_embed
        cls_tokens = self.cls_token.expand(i_tokens.shape[0], -1, -1); i_tokens = torch.cat((cls_tokens, i_tokens), dim=1)
        mask_patch_pool = F.avg_pool2d(motion_mask, kernel_size=self.patch_size, stride=self.patch_size); mask_tokens = mask_patch_pool.flatten(1)
        attention_bias = torch.log(mask_tokens + 1e-6)
        attention_bias = torch.cat([attention_bias, attention_bias], dim=1)
        for block in self.blocks:
            i_tokens, m_tokens, r_tokens = block(i_tokens, m_tokens, r_tokens, attention_bias=attention_bias)
        fused_feature = self.norm(i_tokens[:, 0])
        return self.head(fused_feature)

# --- 时序模块 (使用RoPE并增加了卷积) ---
class RoPE_TransformerEncoderBlock(nn.Module):
    def __init__(self, feature_dim, num_heads, kernel_size=3):
        """
        __init__ 方法已更新，增加了一个 Conv1d 层。
        """
        super().__init__()
        self.norm1 = nn.RMSNorm(feature_dim)
        self.attention = RoPEAttention(feature_dim, num_heads)
        self.norm2 = nn.RMSNorm(feature_dim)
        self.ffn = SwiGLU_FFN(feature_dim)
        
        # --- 新增: 添加一个1D深度可分离卷积层 ---
        # 该层用于捕捉局部的序列模式。
        self.conv = nn.Conv1d(
            in_channels=feature_dim,
            out_channels=feature_dim,
            kernel_size=kernel_size,
            groups=feature_dim,     # 深度可分离卷积
            padding='same',         # 保持序列长度不变
            bias=False
        )
        # --- 新增代码结束 ---

    def forward(self, x, mask=None):
        """
        forward 前向传播函数已更新，以应用新增的卷积层。
        """
        # 1. 自注意力机制 (用于全局关系建模)
        x = x + self.attention(self.norm1(x), self.norm1(x), self.norm1(x), key_padding_mask=mask)
        
        # 2. 前馈网络 (用于特征变换)
        x = x + self.ffn(self.norm2(x))
        
        # --- 新增: 应用卷积层进行局部模式建模 ---
        residual = x
        x = x.permute(0, 2, 1) # (B, S, C) -> (B, C, S)
        x = self.conv(x)
        x = x.permute(0, 2, 1) # (B, C, S) -> (B, S, C)
        x = residual + x
        # --- 新增代码结束 ---
        
        return x

class TeacherTemporalFusion(nn.Module):
    def __init__(self, feature_dim, max_seq_length=20, num_transformer_layers=2, num_transformer_heads=4):
        super(TeacherTemporalFusion, self).__init__()
        # Cross-attention does not use RoPE as Key and Query have different semantic meanings and lengths
        self.cross_attention_fusion = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_transformer_heads, bias=False)
        self.norm_i = nn.RMSNorm(feature_dim)
        self.norm_m = nn.RMSNorm(feature_dim)
        self.transformer_encoder_blocks = nn.ModuleList(
            # --- 注意: 这里需要传递 kernel_size 参数, 如果你想自定义的话 ---
            [RoPE_TransformerEncoderBlock(feature_dim, num_transformer_heads, kernel_size=3) for _ in range(num_transformer_layers)]
        )
    def forward(self, i_features_sequence, motion_summary, video_mask):
        attn_mask = (video_mask == 0)
        # Standard Cross-Attention for fusion
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