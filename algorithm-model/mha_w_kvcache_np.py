import numpy as np
import math
from typing import Tuple, Dict, Any
from numpy.typing import NDArray


class MultiHeadAttentionWithKVCache:
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 2048) -> None:
        """
        Multi-Head Attention with KV Cache implementation
        带KV缓存的多头注意力机制 - 用于推理加速
        
        Args:
            d_model: dimension of model (模型维度)
            num_heads: number of attention heads (注意力头数)
            max_seq_len: maximum sequence length for cache (缓存的最大序列长度)
        """
        # 【面试考点1】：维度校验 - d_model必须能被num_heads整除
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_seq_len = max_seq_len
        
        # 权重矩阵初始化
        self.W_q: NDArray[np.float64] = np.random.randn(d_model, d_model) * 0.01
        self.W_k: NDArray[np.float64] = np.random.randn(d_model, d_model) * 0.01
        self.W_v: NDArray[np.float64] = np.random.randn(d_model, d_model) * 0.01
        self.W_o: NDArray[np.float64] = np.random.randn(d_model, d_model) * 0.01
        
        # 【面试考点2】：KV Cache初始化
        # 预分配缓存空间，避免动态分配带来的性能开销
        self.k_cache = None  # 将在第一次使用时初始化
        self.v_cache = None
        self.cache_len: int = 0   # 当前缓存的序列长度
        
    def init_cache(self, batch_size: int) -> None:
        """
        初始化KV缓存
        【面试考点3】：缓存空间预分配策略
        """
        self.k_cache = np.zeros((batch_size, self.num_heads, self.max_seq_len, self.d_k))
        self.v_cache = np.zeros((batch_size, self.num_heads, self.max_seq_len, self.d_k))
        self.cache_len = 0
    
    def update_cache(self, k: NDArray[np.float64], v: NDArray[np.float64], start_pos: int = None) -> None:
        """
        更新KV缓存
        【面试考点4】：高效的缓存更新机制
        
        Args:
            k: Key tensor (batch_size, num_heads, seq_len, d_k)
            v: Value tensor (batch_size, num_heads, seq_len, d_k)
            start_pos: 缓存的起始位置，用于增量更新
        """
        batch_size, num_heads, seq_len, d_k = k.shape
        
        # 如果缓存未初始化，先初始化
        if self.k_cache is None:
            self.init_cache(batch_size)
        
        # 【面试考点5】：增量更新 vs 全量更新
        
        if start_pos is None:
            # 全量更新：直接替换整个缓存
            self.k_cache[:, :, :seq_len, :] = k
            self.v_cache[:, :, :seq_len, :] = v
            self.cache_len = seq_len
        else:
            # 增量更新：只更新新的token（推理时常用）
            end_pos = start_pos + seq_len
            self.k_cache[:, :, start_pos:end_pos, :] = k
            self.v_cache[:, :, start_pos:end_pos, :] = v
            self.cache_len = max(self.cache_len, end_pos)
    
    def get_cached_kv(self):
        """
        获取缓存的K,V
        【面试考点6】：缓存数据的高效访问
        """
        if self.k_cache is None:
            return None, None
        
        # 只返回有效长度的缓存
        k_cached = self.k_cache[:, :, :self.cache_len, :]
        v_cached = self.v_cache[:, :, :self.cache_len, :]
        return k_cached, v_cached
    
    def clear_cache(self) -> None:
        """
        清空缓存
        【面试考点7】：缓存生命周期管理
        """
        self.k_cache = None
        self.v_cache = None
        self.cache_len = 0
    
    def scaled_dot_product_attention(self, Q: NDArray[np.float64], K: NDArray[np.float64], V: NDArray[np.float64], mask: NDArray[np.float64] = None) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Scaled Dot-Product Attention with efficient computation
        优化的缩放点积注意力计算
        """
        d_k = Q.shape[-1]
        
        # 注意力分数计算
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / math.sqrt(d_k)
        
        # Mask处理
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        
        # Softmax归一化
        attention_weights = self.softmax(scores)
        
        # 加权求和
        output = np.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def softmax(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """数值稳定的softmax实现"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def forward(self, x: NDArray[np.float64], mask: NDArray[np.float64] = None, use_cache: bool = False, start_pos: int = None) -> NDArray[np.float64]:
        """
        Forward pass with optional KV caching
        支持KV缓存的前向传播
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask
            use_cache: 是否使用KV缓存
            start_pos: 缓存起始位置（增量推理时使用）
            
        Returns:
            output: Multi-head attention output
        """
        batch_size, seq_len, d_model = x.shape
        
        # 生成Q,K,V
        Q = np.matmul(x, self.W_q)
        K = np.matmul(x, self.W_k)
        V = np.matmul(x, self.W_v)
        
        # 多头分割
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        if use_cache:
            # 【面试考点8】：KV缓存的使用策略
            if start_pos is not None:
                # 增量推理：更新缓存并使用历史K,V
                self.update_cache(K, V, start_pos)
                K_cached, V_cached = self.get_cached_kv()
                if K_cached is not None:
                    K, V = K_cached, V_cached
            else:
                # 预填充阶段：直接更新缓存
                self.update_cache(K, V)
        
        # 计算注意力
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 多头拼接
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.d_model
        )
        
        # 最终线性变换
        output = np.matmul(attention_output, self.W_o)
        
        return output
    
    def forward_incremental(self, x: NDArray[np.float64], position: int) -> NDArray[np.float64]:
        """
        增量推理的前向传播
        【面试考点9】：高效的增量推理实现
        
        Args:
            x: 单个token的输入 (batch_size, 1, d_model)
            position: 当前token在序列中的位置
        """
        return self.forward(x, use_cache=True, start_pos=position)
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        获取缓存状态信息
        【面试考点10】：缓存状态监控
        """
        if self.k_cache is None:
            return {"cache_initialized": False, "cache_len": 0, "max_len": self.max_seq_len}
        
        cache_size_mb = (self.k_cache.nbytes + self.v_cache.nbytes) / (1024 * 1024)
        return {
            "cache_initialized": True,
            "cache_len": self.cache_len,
            "max_len": self.max_seq_len,
            "cache_size_mb": cache_size_mb,
            "utilization": self.cache_len / self.max_seq_len
        }


def test_mha_with_kv_cache() -> None:
    """
    Test function for Multi-Head Attention with KV Cache
    【面试考点11】：KV缓存功能的完整测试
    """
    print("=== Multi-Head Attention with KV Cache 测试 ===")
    
    # 测试参数
    batch_size = 2
    seq_len = 4
    d_model = 512
    num_heads = 8
    max_seq_len = 1024
    
    print(f"配置参数: d_model={d_model}, num_heads={num_heads}, max_seq_len={max_seq_len}")
    
    # 初始化模型
    mha_cache = MultiHeadAttentionWithKVCache(d_model, num_heads, max_seq_len)
    
    # 测试1：基础功能（不使用缓存）
    print(f"\n=== 测试1: 基础功能 ===")
    x = np.random.randn(batch_size, seq_len, d_model)
    output1 = mha_cache.forward(x, use_cache=False)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output1.shape}")
    assert output1.shape == x.shape, "基础功能测试失败"
    print("✅ 基础功能测试通过")
    
    # 测试2：预填充阶段（使用缓存）
    print(f"\n=== 测试2: 预填充阶段 ===")
    mha_cache.clear_cache()
    output2 = mha_cache.forward(x, use_cache=True)
    cache_info = mha_cache.get_cache_info()
    print(f"缓存状态: {cache_info}")
    print(f"输出一致性: {np.allclose(output1, output2, rtol=1e-5)}")
    assert output2.shape == x.shape, "预填充测试失败"
    print("✅ 预填充阶段测试通过")
    
    # 测试3：增量推理
    print(f"\n=== 测试3: 增量推理 ===")
    # 模拟新token
    new_token = np.random.randn(batch_size, 1, d_model)
    output_incremental = mha_cache.forward_incremental(new_token, position=seq_len)
    
    # 验证缓存更新
    cache_info_after = mha_cache.get_cache_info()
    print(f"增量推理后缓存长度: {cache_info_after['cache_len']}")
    print(f"预期缓存长度: {seq_len + 1}")
    assert cache_info_after['cache_len'] == seq_len + 1, "增量推理缓存更新失败"
    print("✅ 增量推理测试通过")
    
    # 测试4：性能对比模拟
    print(f"\n=== 测试4: 性能分析 ===")
    long_seq = np.random.randn(1, 100, d_model)
    
    # 不使用缓存的计算
    mha_cache.clear_cache()
    output_no_cache = mha_cache.forward(long_seq, use_cache=False)
    
    # 使用缓存的计算
    mha_cache.clear_cache()
    output_with_cache = mha_cache.forward(long_seq, use_cache=True)
    
    print(f"结果一致性: {np.allclose(output_no_cache, output_with_cache, rtol=1e-5)}")
    
    # 【面试考点12】：KV缓存的优势分析
    print(f"\n=== KV缓存优势分析 ===")
    print("1. 内存复用: K,V矩阵只计算一次，后续推理直接使用")
    print("2. 计算加速: 避免重复计算历史token的K,V")
    print("3. 序列扩展: 支持增量添加新token而不重算全序列")
    print("4. 内存效率: 预分配固定大小避免动态扩容")
    
    # 【面试考点13】：实际应用场景
    print(f"\n=== 实际应用场景 ===")
    print("1. 文本生成: GPT类模型的自回归生成")
    print("2. 对话系统: 多轮对话的上下文保持")
    print("3. 代码补全: IDE中的实时代码建议")
    print("4. 机器翻译: 长文本的逐步翻译")
    
    print(f"\n✅ Multi-Head Attention with KV Cache 所有测试通过!")
    print("💡 面试重点: KV缓存是推理优化的关键技术，能显著提升生成任务的效率")


if __name__ == "__main__":
    test_mha_with_kv_cache()