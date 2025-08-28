# file: modules/teacher_modules.py (Flash Attention 数值稳定性修正版)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- RoPE的核心实现 (未修改) ---
class RotaryPositionalEmbedding(nn.Module):
    # ... (这部分代码保持不变)
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
    # ... (这部分代码保持不变)
    q_half1, q_half2 = q.chunk(2, dim=-1)
    k_half1, k_half2 = k.chunk(2, dim=-1)
    q_rotated = torch.cat(
        (q_half1 * cos - q_half2 * sin, q_half1 * sin + q_half2 * cos), dim=-1
    )
    k_rotated = torch.cat(
        (k_half1 * cos - k_half2 * sin, k_half1 * sin + k_half2 * cos), dim=-1
    )
    return q_rotated, k_rotated

# --- 【核心修改 A】: 在 RoPEAttention 中使用数值稳定的浮点数掩码 ---
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
        B, S_q, _ = query.shape
        B, S_kv, _ = key.shape

        q = self.q_proj(query).view(B, S_q, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, S_kv, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, S_kv, self.nhead, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rotary_emb(S_q)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # 【修正】: 将布尔掩码转换为浮点数加性掩码
        attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask shape: [B, S_kv]
            # 我们需要一个可以和 [B, H, S_q, S_kv] 广播的掩码
            attn_mask = torch.zeros(B, 1, 1, S_kv, device=q.device, dtype=q.dtype)
            attn_mask.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -torch.inf)

        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        
        attn_output = attn_output.transpose(1, 2).reshape(B, S_q, -1)
        return self.out_proj(attn_output)

# --- CustomCrossAttention (同样应用浮点数掩码以保持一致性和健壮性) ---
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
        B, S_q, _ = query.shape
        B, S_kv, _ = key.shape
        
        q = self.q_proj(query).view(B, S_q, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, S_kv, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, S_kv, self.nhead, self.head_dim).transpose(1, 2)
        
        # 注意: attention_bias 已经是浮点数，只需确保维度正确
        attn_mask = None
        if attention_bias is not None:
            attn_mask = attention_bias.unsqueeze(1).unsqueeze(2)

        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        
        attn_output = attn_output.transpose(1, 2).reshape(B, S_q, -1)
        return self.out_proj(attn_output)

# --- 【核心修改 B】: 在 FlashSelfAttention 中使用数值稳定的浮点数掩码 ---
class FlashSelfAttention(nn.Module):
    def __init__(self, embed_dim, nhead, bias=False, batch_first=True):
        super().__init__()
        if not batch_first:
            raise NotImplementedError("FlashSelfAttention only supports batch_first=True")
        
        self.nhead = nhead
        self.head_dim = embed_dim // nhead
        assert self.head_dim * nhead == embed_dim, "embed_dim must be divisible by nhead"
        
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, x, key_padding_mask=None):
        B, S, E = x.shape
        
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        
        q = q.view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        
        # 【修正】: 将布尔掩码转换为浮点数加性掩码
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = torch.zeros(B, 1, 1, S, device=q.device, dtype=q.dtype)
            attn_mask.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -torch.inf)
            
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        
        attn_output = attn_output.transpose(1, 2).reshape(B, S, E)
        return self.out_proj(attn_output), None 

# --- 其他模块 (SwiGLU_FFN, MMFT_Block, PatchMerging 等) ---
# --- 无需修改，因为它们的修改是在模块实例化和调用层面上 ---
# ... (以下所有其他模块的代码保持不变)
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
        self.self_attn_i = FlashSelfAttention(embed_dim, nhead, bias=False, batch_first=True)
        self.self_attn_m = FlashSelfAttention(embed_dim, nhead, bias=False, batch_first=True)
        self.self_attn_r = FlashSelfAttention(embed_dim, nhead, bias=False, batch_first=True)
        
        self.cross_attn = CustomCrossAttention(embed_dim, nhead) 
        self.norm1_i = nn.RMSNorm(embed_dim); self.norm1_m = nn.RMSNorm(embed_dim); self.norm1_r = nn.RMSNorm(embed_dim)
        self.norm2_i = nn.RMSNorm(embed_dim); self.norm3_i = nn.RMSNorm(embed_dim)
        self.ffn = SwiGLU_FFN(embed_dim)

    def forward(self, i_tokens, m_tokens, r_tokens, attention_bias=None):
        i_attn_out, _ = self.self_attn_i(self.norm1_i(i_tokens)); i_tokens = i_tokens + i_attn_out
        m_attn_out, _ = self.self_attn_m(self.norm1_m(m_tokens)); m_tokens = m_tokens + m_attn_out
        r_attn_out, _ = self.self_attn_r(self.norm1_r(r_tokens)); r_tokens = r_tokens + r_attn_out

        query = self.norm2_i(i_tokens); kv_tokens = torch.cat([m_tokens, r_tokens], dim=1)
        cross_attn_out = self.cross_attn(query, kv_tokens, kv_tokens, attention_bias=attention_bias)
        i_tokens = i_tokens + cross_attn_out
        i_tokens = i_tokens + self.ffn(self.norm3_i(i_tokens))
        return i_tokens, m_tokens, r_tokens

class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        
        x = x.view(B, H, W, C)
        pad_f = (0, 0, 0, W % 2, 0, H % 2)
        x = F.pad(x, pad_f)
        
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)
        return x

class MMFT_Encoder(nn.Module):
    def __init__(self, output_dim=512, img_size=224, patch_size=32, embed_dim=256, nhead=4, num_layers=2):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        self.patch_embed_i = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.patch_embed_m = nn.Conv2d(2, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.patch_embed_r = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        self.stage1_block = MMFT_Block(embed_dim, nhead)
        self.patch_merging = PatchMerging(dim=embed_dim, norm_layer=nn.RMSNorm)
        self.stage2_block = MMFT_Block(embed_dim * 2, nhead * 2)

        self.norm = nn.RMSNorm(embed_dim * 2)
        self.head = nn.Linear(embed_dim * 2, output_dim)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.RMSNorm):
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, i_frame_image, mv, res, motion_mask):
        B = i_frame_image.shape[0]
        H_patch = i_frame_image.shape[2] // self.patch_size
        W_patch = i_frame_image.shape[3] // self.patch_size

        i_tokens = self.patch_embed_i(i_frame_image).flatten(2).transpose(1, 2)
        m_tokens = self.patch_embed_m(mv).flatten(2).transpose(1, 2)
        r_tokens = self.patch_embed_r(res).flatten(2).transpose(1, 2)
        
        i_tokens = i_tokens + self.pos_embed
        m_tokens = m_tokens + self.pos_embed
        r_tokens = r_tokens + self.pos_embed

        mask_patch_pool = F.avg_pool2d(motion_mask, kernel_size=self.patch_size, stride=self.patch_size)
        
        mask_tokens = mask_patch_pool.flatten(1)
        
        attention_bias = torch.log(mask_tokens + 1e-6)
        attention_bias = torch.cat([attention_bias, attention_bias], dim=1)
        
        i_tokens, m_tokens, r_tokens = self.stage1_block(i_tokens, m_tokens, r_tokens, attention_bias=attention_bias)
        
        i_tokens = self.patch_merging(i_tokens, H_patch, W_patch)
        m_tokens = self.patch_merging(m_tokens, H_patch, W_patch)
        r_tokens = self.patch_merging(r_tokens, H_patch, W_patch)
        
        i_tokens, _, _ = self.stage2_block(i_tokens, m_tokens, r_tokens, attention_bias=None)
        
        fused_feature = self.norm(i_tokens.mean(dim=1))
        return self.head(fused_feature)

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
        self.cross_attention_fusion = CustomCrossAttention(embed_dim=feature_dim, nhead=num_transformer_heads)
        
        self.norm_i = nn.RMSNorm(feature_dim)
        self.norm_m = nn.RMSNorm(feature_dim)
        self.transformer_encoder_blocks = nn.ModuleList(
            [RoPE_TransformerEncoderBlock(feature_dim, num_transformer_heads, kernel_size=3) for _ in range(num_transformer_layers)]
        )
    def forward(self, i_features_sequence, motion_summary, video_mask):
        attn_mask = (video_mask == 0)

        query = self.norm_i(i_features_sequence)
        key = self.norm_m(motion_summary)
        value = self.norm_m(motion_summary)

        cross_attn_out = self.cross_attention_fusion(query, key, value)
        
        enriched_sequence_tensor = i_features_sequence + cross_attn_out
        
        sequence_output = enriched_sequence_tensor * video_mask.unsqueeze(-1)
        for block in self.transformer_encoder_blocks:
            sequence_output = block(sequence_output, mask=attn_mask)
        return sequence_output * video_mask.unsqueeze(-1)