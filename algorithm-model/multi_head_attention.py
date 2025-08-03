import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        Multi-Head Attention implementation
        多头注意力机制实现 - Transformer核心组件
        
        Args:
            d_model: dimension of model (模型维度)
            num_heads: number of attention heads (注意力头数)
        """
        super(MultiHeadAttention, self).__init__()
        
        # 【面试考点1】：维度校验 - d_model必须能被num_heads整除
        # 这是因为每个头要分配相等的维度 d_k = d_model // num_heads
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # 【面试考点2】：参数初始化策略
        # 使用PyTorch的nn.Linear层，自动进行权重初始化
        # PyTorch默认使用Kaiming uniform初始化
        self.W_q = nn.Linear(d_model, d_model, bias=False)  # Query投影矩阵
        self.W_k = nn.Linear(d_model, d_model, bias=False)  # Key投影矩阵  
        self.W_v = nn.Linear(d_model, d_model, bias=False)  # Value投影矩阵
        self.W_o = nn.Linear(d_model, d_model, bias=False)  # 输出投影矩阵
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Scaled Dot-Product Attention - 注意力机制的核心计算
        缩放点积注意力：Attention(Q,K,V) = softmax(QK^T/√d_k)V
        
        Args:
            Q: Query matrix (batch_size, num_heads, seq_len, d_k)
            K: Key matrix (batch_size, num_heads, seq_len, d_k) 
            V: Value matrix (batch_size, num_heads, seq_len, d_k)
            mask: Optional mask matrix (用于处理padding或因果mask)
            
        Returns:
            output: Attention output
            attention_weights: Attention weights
        """
        d_k = Q.shape[-1]
        
        # 【面试考点3】：注意力分数计算 - QK^T
        # 矩阵乘法：(batch, heads, seq_len, d_k) × (batch, heads, d_k, seq_len)
        # 结果形状：(batch, heads, seq_len, seq_len) - 每个位置对所有位置的注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 【面试考点4】：缩放因子√d_k的作用
        # 防止softmax饱和：当d_k很大时，点积值会很大，导致softmax梯度接近0
        # 除以√d_k可以控制点积的方差，保持梯度稳定
        
        # 【面试考点5】：Mask机制处理
        # 用-inf替换mask位置，softmax后这些位置权重为0
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 【面试考点6】：Softmax归一化
        # 将注意力分数转换为概率分布，保证权重和为1
        attention_weights = F.softmax(scores, dim=-1)
        
        # 【面试考点7】：加权求和
        # 用注意力权重对Value进行加权平均，得到上下文向量
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    
    def forward(self, x, mask=None):
        """
        Forward pass of Multi-Head Attention
        多头注意力前向传播 - 完整的计算流程
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            output: Multi-head attention output
        """
        batch_size, seq_len, d_model = x.shape
        
        # 【面试考点9】：线性投影生成Q,K,V
        # 将输入通过不同的权重矩阵投影得到Query, Key, Value
        # 这使得模型能学习到不同的表示子空间
        Q = self.W_q(x)  # (batch_size, seq_len, d_model)
        K = self.W_k(x)  # (batch_size, seq_len, d_model)
        V = self.W_v(x)  # (batch_size, seq_len, d_model)
        
        # 【面试考点10】：多头分割 - 核心的维度变换
        # 将d_model维度分割成num_heads个d_k维度的子空间
        # reshape: (batch, seq_len, d_model) -> (batch, seq_len, num_heads, d_k)
        # permute: (batch, seq_len, num_heads, d_k) -> (batch, num_heads, seq_len, d_k)
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        
        # 【面试考点11】：并行计算多个注意力头
        # 每个头独立计算注意力，捕获不同的语义关系
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 【面试考点12】：多头拼接 - 恢复原始维度
        # permute: (batch, num_heads, seq_len, d_k) -> (batch, seq_len, num_heads, d_k)
        # reshape: (batch, seq_len, num_heads, d_k) -> (batch, seq_len, d_model)
        attention_output = attention_output.permute(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.d_model
        )
        
        # 【面试考点13】：最终线性变换
        # 通过输出投影矩阵W_o整合多头信息，这是可学习的参数
        output = self.W_o(attention_output)
        
        return output


def test_multi_head_attention():
    """
    Test function for Multi-Head Attention
    【面试考点14】：完整的测试用例设计
    """
    print("=== Multi-Head Attention 算法工程师面试测试 ===")
    
    # 【面试考点15】：典型的超参数设置
    batch_size = 2      # 批大小
    seq_len = 4         # 序列长度  
    d_model = 512       # 模型维度 (Transformer标准配置)
    num_heads = 8       # 注意力头数 (Transformer标准配置)
    
    print(f"配置参数: d_model={d_model}, num_heads={num_heads}")
    print(f"每个头的维度 d_k = d_model // num_heads = {d_model // num_heads}")
    
    # Create test input
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"输入形状: {x.shape}")
    
    # Initialize multi-head attention
    mha = MultiHeadAttention(d_model, num_heads)
    mha.eval()  # Set to evaluation mode
    
    # 【面试考点16】：计算复杂度分析
    print(f"\n=== 复杂度分析 ===")
    print(f"时间复杂度: O(n²·d) 其中n={seq_len}, d={d_model}")
    print(f"空间复杂度: O(n²·h) 其中h={num_heads} (存储attention矩阵)")
    
    # Forward pass
    output = mha.forward(x)
    
    print(f"\n=== 输出验证 ===")
    print(f"输出形状: {output.shape}")
    print(f"形状是否保持: {tuple(output.shape) == tuple(x.shape)}")
    
    # 【面试考点17】：维度校验 - 关键的正确性检查
    assert output.shape == x.shape, f"形状不匹配! 期望{x.shape}, 得到{output.shape}"
    
    # 【面试考点18】：数值范围检查
    print(f"输出数值范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"输出均值: {output.mean().item():.4f}, 标准差: {output.std().item():.4f}")
    
    # 【面试考点19】：梯度检查 (简化版)
    # 在实际面试中可能会要求实现反向传播
    print(f"\n=== 关键设计要点总结 ===")
    print("1. 维度变换: (batch,seq,d_model) -> (batch,heads,seq,d_k)")
    print("2. 缩放因子: 1/√d_k 防止softmax饱和") 
    print("3. 并行计算: 多头独立计算后拼接")
    print("4. 残差连接: 实际应用中需要加上输入 (未在此实现)")
    print("5. Layer Norm: 通常在attention后添加 (未在此实现)")
    
    print("\n✅ Multi-Head Attention 所有测试通过!")
    print("💡 面试重点: 能解释每个步骤的数学原理和工程实现细节")


if __name__ == "__main__":
    test_multi_head_attention()