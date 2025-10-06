import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import loralib as lora
import complex.complex_module as cm
from einops import rearrange, repeat
import numpy as np
from .mimo_model import (DiffusionEmbedding, MLPConditionEmbedding, 
                        PositionEmbedding, init_weight_norm, 
                        init_weight_xavier, init_weight_zero)

@torch.jit.script
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class ComplexMultiHeadAttentionLoRA(cm.ComplexMultiHeadAttention):
    """ComplexMultiHeadAttention with LoRA adaptation"""
    def __init__(self, in_dim, out_dim, num_heads, dropout, bias, r, 
                 lora_alpha, lora_dropout, **kwargs):
        super().__init__(in_dim, out_dim, num_heads, dropout, bias=bias, **kwargs)
        
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

class DiAWithLoRA(nn.Module):
    """DiA block with LoRA adaptation for MIMO model (dual attention structure)"""
    def __init__(self, hidden_dim, num_heads, dropout, mlp_ratio=4.0, 
                 lora_r=8, lora_alpha=16, lora_dropout=0.1):
        super().__init__()
        self.norm1 = cm.NaiveComplexLayerNorm(
            hidden_dim, eps=1e-6, elementwise_affine=False)
            
        # Self attention with LoRA
        self.s_attn = ComplexMultiHeadAttentionLoRA(
            hidden_dim, hidden_dim, num_heads, dropout, bias=True,
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
        )
        
        self.norm2 = cm.NaiveComplexLayerNorm(
            hidden_dim, eps=1e-6, elementwise_affine=False)
        self.normc = cm.NaiveComplexLayerNorm(
            hidden_dim, eps=1e-6, elementwise_affine=False)
            
        # Cross attention with LoRA
        self.x_attn = cm.ComplexMultiHeadAttention(
            hidden_dim, hidden_dim, num_heads, dropout, bias=True)
        
        self.norm3 = cm.NaiveComplexLayerNorm(
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
        self.adaLN_modulation.apply(init_weight_zero)

    def forward(self, x, t, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            t).chunk(6, dim=1)
            
        # Self attention path
        mod_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.s_attn(mod_x, mod_x, mod_x)
        
        # Cross attention path
        x = x + self.x_attn(
            queries=self.normc(c),
            keys=self.norm2(x),
            values=self.norm2(x)
        )
        
        # MLP path
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x

class SpatialDiffusionLoRA(nn.Module):
    """SpatialDiffusion with LoRA adaptation"""
    def __init__(self, params):
        super().__init__()
        self.learn_tfdiff = params.learn_tfdiff
        self.num_block = params.num_spatial_block
        self.input_dim = params.extra_dim[-1]
        self.input_len = params.extra_dim[-2]
        self.output_dim = self.input_dim * self.input_len
        self.hidden_dim = params.spatial_hidden_dim
        self.num_heads = params.num_heads
        self.max_step = params.max_step
        self.embed_dim = params.embed_dim
        self.cond_dim = params.cond_dim[-1]
        self.dropout = params.dropout
        self.task_id = params.task_id
        self.mlp_ratio = params.mlp_ratio
        
        # LoRA parameters
        self.lora_r = getattr(params, 'lora_r', 8)
        self.lora_alpha = getattr(params, 'lora_alpha', 1)
        self.lora_dropout = getattr(params, 'lora_dropout', 0.)
        
        self.p_embed = PositionEmbedding(
            self.input_len, self.input_dim, self.hidden_dim)
        self.t_embed = DiffusionEmbedding(
            self.max_step, self.embed_dim, self.hidden_dim)
        self.c_embed = MLPConditionEmbedding(self.cond_dim, self.hidden_dim)
        
        # DiA blocks with LoRA
        self.blocks = nn.ModuleList([
            DiAWithLoRA(
                self.hidden_dim, self.num_heads, self.dropout, self.mlp_ratio,
                lora_r=self.lora_r, lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout
            ) for _ in range(self.num_block)
        ])
        
        self.adaMLP = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-2),
            cm.ComplexLinear(self.input_len*self.hidden_dim, self.output_dim),
            cm.ComplexSiLU(),
            cm.ComplexLinear(self.output_dim, self.output_dim),
        )
        self.adaMLP.apply(init_weight_xavier)

    def forward(self, x, t, c):
        x = self.p_embed(x)
        t = self.t_embed(t)
        c = self.c_embed(c)
        for block in self.blocks:
            x = block(x, t, c)
        x = self.adaMLP(x)
        return x

class TimeFrequencyDiffusionLoRA(nn.Module):
    """TimeFrequencyDiffusion with LoRA adaptation"""
    def __init__(self, params):
        super().__init__()
        self.learn_tfdiff = params.learn_tfdiff
        self.num_block = params.num_tf_block
        self.input_dim = np.prod(params.extra_dim)
        self.input_len = params.sample_rate
        self.output_dim = self.input_dim
        self.hidden_dim = params.tf_hidden_dim
        self.num_heads = params.num_heads
        self.max_step = params.max_step
        self.embed_dim = params.embed_dim
        self.cond_dim = np.prod(params.cond_dim)
        self.dropout = params.dropout
        self.task_id = params.task_id
        self.mlp_ratio = params.mlp_ratio
        
        # LoRA parameters
        self.lora_r = getattr(params, 'lora_r', 8)
        self.lora_alpha = getattr(params, 'lora_alpha', 16)
        self.lora_dropout = getattr(params, 'lora_dropout', 0.1)
        
        self.p_embed = PositionEmbedding(
            self.input_len, self.input_dim, self.hidden_dim)
        self.t_embed = DiffusionEmbedding(
            self.max_step, self.embed_dim, self.hidden_dim)
        self.c_embed = MLPConditionEmbedding(self.cond_dim, self.hidden_dim)
        
        # DiA blocks with LoRA
        self.blocks = nn.ModuleList([
            DiAWithLoRA(
                self.hidden_dim, self.num_heads, self.dropout, self.mlp_ratio,
                lora_r=self.lora_r, lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout
            ) for _ in range(self.num_block)
        ])
        
        self.final_layer = FinalLayer(
            self.hidden_dim, self.output_dim)

    def forward(self, x, t, c):
        x = self.p_embed(x)
        t = self.t_embed(t)
        # c = c.reshape([-1, self.input_len, 2496, 2])
        c = rearrange(c, 'B input_len subcarr ant complex -> B input_len (subcarr ant) complex')
        c = self.c_embed(c)
        for block in self.blocks:
            x = block(x, t, c)
        x = self.final_layer(x, t)
        return x

class tfdiff_mimo_LoRA(nn.Module):
    """MIMO model with LoRA adaptation"""
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.task_id = params.task_id
        self.sample_rate = params.sample_rate
        self.extra_dim = params.extra_dim
        self.cond_dim = params.cond_dim
        self.spatial_dim = np.prod(self.extra_dim)
        
        # Using LoRA versions of components
        self.spatial_block = SpatialDiffusionLoRA(self.params)
        self.tf_block = TimeFrequencyDiffusionLoRA(self.params)

    def forward(self, x, t, c):
        x_s = rearrange(x, 'B sample_rate subcarrier ant complex -> (B sample_rate) subcarrier ant complex')
        # x_s = x.reshape([-1]+self.extra_dim+[2])  # [B*N, S, A, 2] 
        c_s = rearrange(c, 'B sample_rate subcarrier ant complex -> (B sample_rate) subcarrier ant complex')
        # c_s = c.reshape([-1]+self.cond_dim+[2])  # [B*N, [C], 2]
        t_s = repeat(t, 'B -> (B N)', N=self.sample_rate)
        # TODO:repeat funtion??
        x_s = self.spatial_block(x_s, t_s, c_s)  # [B*N, S*A, 2]
        # x = rearrange(x_s, '(B sample_rate) subcarrier ant complex -> B sample_rate (subcarrier ant) complex', sample_rate=self.sample_rate)
        x = rearrange(x_s, '(B sample_rate) spatial complex -> B sample_rate spatial complex', sample_rate=self.sample_rate)
        # x = x_s.reshape([-1, self.sample_rate] +
        #                 [self.spatial_dim, 2])  # [B, N, S*A, 2]
        x = self.tf_block(x, t, c)  # [B, N, S*A, 2]
        x = rearrange(x, 'B sample_rate (subcarrier ant) complex -> B sample_rate subcarrier ant complex', sample_rate=self.sample_rate, subcarrier=self.extra_dim[0], ant=self.extra_dim[1])
        # x = x.reshape([-1, self.sample_rate] +
        #               self.extra_dim+[2])  # [B, N, S, A, 2]
        return x

class FinalLayer(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.norm = cm.NaiveComplexLayerNorm(
            hidden_dim, eps=1e-6, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            cm.ComplexSiLU(),
            cm.ComplexLinear(hidden_dim, 2*hidden_dim, bias=True)
        )
        self.linear = cm.ComplexLinear(hidden_dim, out_dim, bias=True)
        self.apply(init_weight_zero)

    def forward(self, x, t):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = x + modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x
