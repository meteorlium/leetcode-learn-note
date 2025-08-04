import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GroupQueryAttention(nn.Module):
    def __init__(self, d_model, num_query_heads, num_kv_heads):
        """
        Group Query Attention implementation
        分组查询注意力机制实现 - 高效推理的Transformer组件
        
        Args:
            d_model: dimension of model (模型维度)
            num_query_heads: number of query heads (查询头数)
            num_kv_heads: number of key-value heads (键值头数)
        """
        super(GroupQueryAttention, self).__init__()
        
        # 【面试考点1】：GQA核心约束 - Query头数必须是KV头数的整数倍
        # 这样才能将Query头均匀分组，每组共享一个KV头
        assert num_query_heads % num_kv_heads == 0, "num_query_heads必须是num_kv_heads的整数倍"
        assert d_model % num_query_heads == 0, "d_model必须能被num_query_heads整除"
        
        self.d_model = d_model
        self.num_query_heads = num_query_heads  # Query头数（通常较多）
        self.num_kv_heads = num_kv_heads        # KV头数（通常较少）
        self.d_k = d_model // num_query_heads   # 每个Query头的维度
        self.d_kv = d_model // num_kv_heads     # 每个KV头的维度
        
        # 【面试考点2】：分组比例计算
        # group_size表示多少个Query头共享一个KV头
        self.group_size = num_query_heads // num_kv_heads
        
        # 【面试考点3】：参数量对比分析
        # Query投影：保持原有维度 (d_model -> d_model)
        # KV投影：维度减少 (d_model -> num_kv_heads * d_kv)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.d_kv, bias=False)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.d_kv, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Scaled Dot-Product Attention for GQA
        GQA版本的缩放点积注意力：处理不同头数的Q和KV
        
        Args:
            Q: Query matrix (batch_size, num_query_heads, seq_len, d_k)
            K: Key matrix (batch_size, num_kv_heads, seq_len, d_kv) 
            V: Value matrix (batch_size, num_kv_heads, seq_len, d_kv)
            mask: Optional mask matrix
            
        Returns:
            output: Attention output
            attention_weights: Attention weights
        """
        d_k = Q.shape[-1]
        
        # 【面试考点4】：KV头复制策略
        # 将每个KV头复制group_size次，使其匹配Query头数
        # repeat_interleave保证分组对应关系正确
        K = K.repeat_interleave(self.group_size, dim=1)  # (batch, num_query_heads, seq_len, d_kv)
        V = V.repeat_interleave(self.group_size, dim=1)  # (batch, num_query_heads, seq_len, d_kv)

        """
        interleave = 交错、穿插
        - 像编织一样，将元素交替排列

        repeat_interleave = 就地重复每个元素
        - [1,2,3].repeat_interleave(2) → [1,1,2,2,3,3]
        - [1,2,3].repeat(2) → [1,2,3,1,2,3]
        """
        
        # 【面试考点5】：维度一致性检查
        # 确保Q和扩展后的K,V在头数维度上匹配
        assert Q.shape[1] == K.shape[1] == V.shape[1], "头数维度不匹配"
        
        # 注意力分数计算：QK^T / √d_k
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Mask处理
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax归一化
        attention_weights = F.softmax(scores, dim=-1)
        
        # 加权求和
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    
    def forward(self, x, mask=None):
        """
        Forward pass of Group Query Attention
        分组查询注意力前向传播
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            output: Group query attention output
        """
        batch_size, seq_len, d_model = x.shape
        
        # 【面试考点6】：不对称的线性投影
        # Query维持原有复杂度，KV投影维度减少
        Q = self.W_q(x)  # (batch_size, seq_len, d_model)
        K = self.W_k(x)  # (batch_size, seq_len, num_kv_heads * d_kv)
        V = self.W_v(x)  # (batch_size, seq_len, num_kv_heads * d_kv)
        
        # 【面试考点7】：分组重塑策略
        # Query: 重塑为标准多头格式
        Q = Q.reshape(batch_size, seq_len, self.num_query_heads, self.d_k).permute(0, 2, 1, 3)
        
        # KV: 重塑为较少的头数格式
        K = K.reshape(batch_size, seq_len, self.num_kv_heads, self.d_kv).permute(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_kv_heads, self.d_kv).permute(0, 2, 1, 3)
        
        # 【面试考点8】：分组注意力计算
        # 通过KV头复制实现分组共享机制
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 【面试考点9】：输出拼接恢复
        # 将多头输出拼接回原始维度
        attention_output = attention_output.permute(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.d_model
        )
        
        # 最终线性变换
        output = self.W_o(attention_output)
        
        return output


def test_group_query_attention():
    """
    Test function for Group Query Attention
    【面试考点10】：GQA vs MHA 对比测试
    """
    print("=== Group Query Attention 算法工程师面试测试 ===")
    
    # 【面试考点11】：典型的GQA配置
    batch_size = 2
    seq_len = 4
    d_model = 512
    num_query_heads = 32    # Query头数较多
    num_kv_heads = 8        # KV头数较少，4:1分组比例
    
    print(f"配置参数: d_model={d_model}")
    print(f"Query头数: {num_query_heads}, KV头数: {num_kv_heads}")
    print(f"分组大小: {num_query_heads // num_kv_heads} (每{num_query_heads // num_kv_heads}个Query头共享1个KV头)")
    
    # Create test input
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"输入形状: {x.shape}")
    
    # Initialize group query attention
    gqa = GroupQueryAttention(d_model, num_query_heads, num_kv_heads)
    gqa.eval()
    
    # 【面试考点12】：参数量对比分析
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())
    
    gqa_params = count_parameters(gqa)
    
    # 对比标准MHA的参数量
    from multi_head_attention import MultiHeadAttention
    mha = MultiHeadAttention(d_model, num_query_heads)
    mha_params = count_parameters(mha)
    
    print(f"\n=== 参数量对比 ===")
    print(f"GQA参数量: {gqa_params:,}")
    print(f"MHA参数量: {mha_params:,}")
    print(f"参数减少: {((mha_params - gqa_params) / mha_params * 100):.1f}%")
    
    # 【面试考点13】：计算复杂度分析
    print(f"\n=== 复杂度分析 ===")
    print(f"KV投影复杂度: O(d·n·h_kv) 其中h_kv={num_kv_heads}")
    print(f"注意力计算: O(n²·h_q) 其中h_q={num_query_heads}")
    print(f"KV缓存大小: {num_kv_heads}/{num_query_heads} = {num_kv_heads/num_query_heads:.2f}倍 (相比MHA)")
    
    # Forward pass
    output = gqa.forward(x)
    
    print(f"\n=== 输出验证 ===")
    print(f"输出形状: {output.shape}")
    print(f"形状是否保持: {tuple(output.shape) == tuple(x.shape)}")
    
    # 【面试考点14】：正确性检查
    assert output.shape == x.shape, f"形状不匹配! 期望{x.shape}, 得到{output.shape}"
    
    print(f"输出数值范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"输出均值: {output.mean().item():.4f}, 标准差: {output.std().item():.4f}")
    
    # 【面试考点15】：GQA关键优势总结
    print(f"\n=== GQA核心优势 ===")
    print("1. 参数效率: KV投影参数减少，降低模型大小")
    print("2. 推理加速: KV缓存大小减少，提升长序列推理速度")
    print("3. 质量保持: Query头数不变，保持表达能力")
    print("4. 内存优化: 推理时KV缓存内存占用显著降低")
    print("5. 分组设计: 多个Query头共享KV头，平衡效率与性能")
    
    # 【面试考点16】：适用场景分析
    print(f"\n=== 应用场景 ===")
    print("• 大模型推理优化 (如LLaMA-2, Code Llama)")
    print("• 长序列生成任务 (减少KV缓存压力)")  
    print("• 资源受限环境 (移动端、边缘计算)")
    print("• 实时对话系统 (降低推理延迟)")
    
    print("\n✅ Group Query Attention 所有测试通过!")
    print("💡 面试重点: 理解GQA的分组机制和效率优势")


if __name__ == "__main__":
    test_group_query_attention()