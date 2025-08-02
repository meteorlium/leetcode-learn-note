"""
LLM LoRA (Low-Rank Adaptation) 实现
面试重点：参数高效微调技术，显著降低训练成本
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import math


class LoRALinear(nn.Module):
    """
    LoRA 线性层实现
    
    面试考点：
    1. 为什么用低秩分解？降低参数量和计算量
    2. r 的选择对性能的影响？r 越大表达能力越强但参数越多
    3. alpha 的作用？控制 LoRA 权重的缩放，影响训练稳定性
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,  # 面试重点：低秩参数 r，通常取 4-64
        alpha: float = 16.0,  # 面试重点：缩放因子，通常取 16 或 32
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()
        
        # 原始预训练权重（冻结）
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.linear.weight.requires_grad = False
        if bias:
            self.linear.bias.requires_grad = False
            
        # LoRA 参数：W = W0 + BA，其中 B(d×r), A(r×k)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank  # 面试重点：缩放避免训练不稳定
        
        # 低秩分解矩阵
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))  # (r, d_in)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))  # (d_out, r)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.reset_parameters()
        
    def reset_parameters(self):
        """
        面试考点：初始化策略
        - A 用随机初始化（类似 Kaiming）
        - B 用零初始化，确保初始时 ΔW = 0
        """
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：y = W0*x + (B*A)*x * scaling
        
        面试重点：
        1. 为什么先算 Ax 再算 B(Ax)？减少计算量 O(rank*(d_in+d_out))
        2. scaling 何时应用？在最后，避免梯度消失
        """
        # 原始线性变换
        result = self.linear(x)
        
        # LoRA 增量：先 A 后 B，降低计算复杂度
        if self.dropout is not None:
            x = self.dropout(x)
            
        # 维度详解：
        # x: (batch, seq, d_in), lora_A: (rank, d_in), lora_B: (d_out, rank)
        # 目标：计算 x * A^T * B^T，其中 A^T: (d_in, rank), B^T: (rank, d_out)
        lora_result = x @ self.lora_A.T  # (batch, seq, d_in) @ (d_in, rank) = (batch, seq, rank)
        lora_result = lora_result @ self.lora_B.T  # (batch, seq, rank) @ (rank, d_out) = (batch, seq, d_out)
        
        return result + lora_result * self.scaling


class LoRAAttention(nn.Module):
    """
    对 Multi-Head Attention 应用 LoRA
    
    面试考点 - LoRA 应用策略的演进：
    1. 经典做法（2021 原始论文）：只对 Q、V 做 LoRA，K 保持不变
       - 理由：计算预算限制，Q、V 对输出影响更大
       - Microsoft 原始实验：Q+V 组合效果接近全参数微调
    
    2. 现代最佳实践（2024-2025）：对所有线性层做 LoRA
       - QLoRA 等新研究：Q、K、V、O 全部应用 LoRA 效果更好
       - 实验结论：全线性层 LoRA 比仅 Q+V 有显著提升
       - 参数增加有限但性能提升明显
       
    3. 为什么 K 现在也重要？
       - K 矩阵影响注意力权重分布和查询匹配
       - 低秩扰动对注意力聚类行为有重要影响
       - 现代硬件计算能力提升，参数预算不再是主要限制
    """
    
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 使用 LoRA 的投影层
        self.w_q = LoRALinear(d_model, d_model, lora_rank, lora_alpha, dropout, bias=False)
        # 现代最佳实践：K 也使用 LoRA（原始论文中为了节省计算而跳过）
        self.w_k = LoRALinear(d_model, d_model, lora_rank, lora_alpha, dropout, bias=False)
        self.w_v = LoRALinear(d_model, d_model, lora_rank, lora_alpha, dropout, bias=False)
        self.w_o = LoRALinear(d_model, d_model, lora_rank, lora_alpha, dropout, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        面试重点：LoRA 不改变注意力机制本身，只改变权重矩阵
        """
        batch_size, seq_len = query.size(0), query.size(1)
        
        # 线性投影（包含 LoRA）
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
            
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.w_o(context)


class SimpleLLMWithLoRA(nn.Module):
    """
    简化的 LLM 模型应用 LoRA
    
    面试考点：
    1. 哪些层需要 LoRA？注意力层和 FFN 层
    2. LoRA 的训练策略？只训练 LoRA 参数，冻结原始权重
    3. 推理时如何处理？可以将 LoRA 权重合并到原始权重中
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        max_seq_len: int = 1024,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # 词嵌入（通常不用 LoRA）
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(max_seq_len, d_model))
        
        # Transformer 层（应用 LoRA）
        self.layers = nn.ModuleList([
            TransformerBlockWithLoRA(d_model, n_heads, lora_rank, lora_alpha, dropout)
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        
        # 输出层（可选择性应用 LoRA）
        self.head = LoRALinear(d_model, vocab_size, lora_rank, lora_alpha, bias=False)
        
    def forward(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len = input_ids.size(1)
        
        # 嵌入
        x = self.embedding(input_ids)
        x = x + self.pos_embedding[:seq_len]
        
        # Transformer 层
        for layer in self.layers:
            x = layer(x, mask)
            
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits
    
    def merge_lora_weights(self):
        """
        面试重点：推理优化 - 将 LoRA 权重合并到原始权重
        合并后推理速度与原模型相同
        """
        for module in self.modules():
            if isinstance(module, LoRALinear):
                # W_new = W_original + B @ A * scaling
                with torch.no_grad():
                    delta_w = (module.lora_B @ module.lora_A) * module.scaling
                    module.linear.weight.data += delta_w
                    # 清除 LoRA 参数
                    module.lora_A.data.zero_()
                    module.lora_B.data.zero_()


class TransformerBlockWithLoRA(nn.Module):
    """Transformer Block with LoRA applied to attention and FFN"""
    
    def __init__(self, d_model: int, n_heads: int, lora_rank: int, lora_alpha: float, dropout: float):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = LoRAAttention(d_model, n_heads, lora_rank, lora_alpha, dropout)
        
        self.ln2 = nn.LayerNorm(d_model)
        # FFN 也应用 LoRA
        self.ffn = nn.Sequential(
            LoRALinear(d_model, 4 * d_model, lora_rank, lora_alpha, dropout),
            nn.GELU(),
            LoRALinear(4 * d_model, d_model, lora_rank, lora_alpha, dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 注意力层
        attn_out = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), mask)
        x = x + self.dropout(attn_out)
        
        # FFN 层
        ffn_out = self.ffn(self.ln2(x))
        x = x + self.dropout(ffn_out)
        
        return x


# 面试常见问题示例
def demonstrate_lora_advantages():
    """
    面试重点：LoRA 的优势演示
    """
    print("=== LoRA 参数量对比 ===")
    
    # 原始模型参数量
    d_model, vocab_size = 512, 10000
    original_params = d_model * vocab_size  # 只看一个线性层
    
    # LoRA 参数量
    rank = 8
    lora_params = rank * (d_model + vocab_size)
    
    print(f"原始线性层参数量: {original_params:,}")
    print(f"LoRA 参数量: {lora_params:,}")
    print(f"参数减少比例: {(1 - lora_params/original_params)*100:.1f}%")
    
    # 不同 rank 的影响
    print("\n=== 不同 rank 的参数量 ===")
    for r in [4, 8, 16, 32, 64]:
        lora_p = r * (d_model + vocab_size)
        ratio = lora_p / original_params * 100
        print(f"rank={r}: {lora_p:,} 参数 ({ratio:.2f}% of original)")


if __name__ == "__main__":
    # 面试演示代码
    print("LoRA 实现演示")
    
    # 创建模型
    model = SimpleLLMWithLoRA(
        vocab_size=1000,
        d_model=256,
        n_heads=8,
        n_layers=4,
        lora_rank=8
    )
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"可训练参数比例: {trainable_params/total_params*100:.1f}%")
    
    # 演示前向传播
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    with torch.no_grad():
        output = model(input_ids)
        print(f"输出形状: {output.shape}")
    
    demonstrate_lora_advantages()