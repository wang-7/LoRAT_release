import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import loralib as lora
from .wifi_model import (DiffusionEmbedding, MLPConditionEmbedding, 
                        PositionEmbedding, init_weight_norm, init_weight_xavier)
import complex.complex_module as cm

class DiAWithLoRA(nn.Module):
    """DiA (Diffusion Attention) block with LoRA adaptation"""
    def __init__(self, hidden_dim, num_heads, dropout, mlp_ratio=4.0, 
                 lora_r=8, lora_alpha=16, lora_dropout=0.1):
        super().__init__()
        self.norm1 = cm.NaiveComplexLayerNorm(
            hidden_dim, eps=1e-6, elementwise_affine=False)
            
        # 将原始的 ComplexMultiHeadAttention 替换为带有 LoRA 的版本
        self.attn = ComplexMultiHeadAttentionLoRA(
            hidden_dim, hidden_dim, num_heads, dropout, bias=True,
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
        )
        
        self.norm2 = cm.NaiveComplexLayerNorm(
            hidden_dim, eps=1e-6, elementwise_affine=False)
            
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            cm.ComplexLinear(hidden_dim, mlp_hidden_dim, bias=True),
            cm.ComplexSiLU(),
            cm.ComplexLinear(mlp_hidden_dim, hidden_dim, bias=True),
        )
        
        self.adaLN_modulation = nn.Sequential(
            cm.ComplexSiLU(),
            cm.ComplexLinear(hidden_dim, 6*hidden_dim, bias=True)
        )
        
        self.apply(init_weight_xavier)
        self.adaLN_modulation.apply(init_weight_norm)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c).chunk(6, dim=1)
            
        # 使用带有 LoRA 的注意力层
        mod_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(mod_x, mod_x, mod_x)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class ComplexMultiHeadAttentionLoRA(cm.ComplexMultiHeadAttention):
    """ComplexMultiHeadAttention with LoRA adaptation"""
    def __init__(self, in_dim, out_dim, num_heads, dropout, bias, r=0, 
                 lora_alpha=1, lora_dropout=0., **kwargs):
        super().__init__(in_dim, out_dim, num_heads, dropout, bias=bias, **kwargs)
        
        # 替换原始的 query, key, value 投影为 LoRA 版本
        self.w_q = lora.ComplexLinear(
            in_dim, out_dim, bias=bias,
            r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
        )
        self.w_k = lora.ComplexLinear(
            in_dim, out_dim, bias=bias,
            r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
        )
        self.w_v = lora.ComplexLinear(
            in_dim, out_dim, bias=bias,
            r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
        )

class tfdiff_WiFi_LoRA(nn.Module):
    """WiFi model with LoRA adaptation"""
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.learn_tfdiff = params.learn_tfdiff
        self.input_dim = params.input_dim
        self.output_dim = self.input_dim
        self.hidden_dim = params.hidden_dim
        self.num_heads = params.num_heads
        self.dropout = params.dropout
        self.task_id = params.task_id
        self.mlp_ratio = params.mlp_ratio
        
        # LoRA 配置
        self.lora_r = getattr(params, 'lora_r', 8)
        self.lora_alpha = getattr(params, 'lora_alpha', 1)
        self.lora_dropout = getattr(params, 'lora_dropout', 0.1)
        
        # 保持原有的嵌入层不变
        self.p_embed = PositionEmbedding(
            params.sample_rate, params.input_dim, params.hidden_dim)
        self.t_embed = DiffusionEmbedding(
            params.max_step, params.embed_dim, params.hidden_dim)
        self.c_embed = MLPConditionEmbedding(params.cond_dim, params.hidden_dim)
        
        # 使用带有 LoRA 的注意力块
        self.blocks = nn.ModuleList([
            DiAWithLoRA(
                self.hidden_dim, self.num_heads, self.dropout, self.mlp_ratio,
                lora_r=self.lora_r, lora_alpha=self.lora_alpha, 
                lora_dropout=self.lora_dropout
            ) for _ in range(params.num_block)
        ])
        
        self.final_layer = FinalLayer(self.hidden_dim, self.output_dim)

    def forward(self, x, t, c):
        x = self.p_embed(x)
        t = self.t_embed(t)
        c = self.c_embed(c)
        c = c + t
        
        for block in self.blocks:
            x = block(x, c)
            
        x = self.final_layer(x, c)
        return x

# 辅助函数
@torch.jit.script
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class FinalLayer(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.norm = cm.NaiveComplexLayerNorm(
            hidden_dim, eps=1e-6, elementwise_affine=False)
        self.linear = cm.ComplexLinear(hidden_dim, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            cm.ComplexSiLU(),
            cm.ComplexLinear(hidden_dim, 2*hidden_dim, bias=True)
        )
        self.apply(init_weight_norm)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x