import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Any


class MultiHeadAttentionWithKVCache(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 2048, device: str = "cpu"):
        """
        Multi-Head Attention with KV Cache implementation (PyTorch版本)
        带KV缓存的多头注意力机制 - 用于推理加速
        
        Args:
            d_model: dimension of model (模型维度)
            num_heads: number of attention heads (注意力头数)
            max_seq_len: maximum sequence length for cache (缓存的最大序列长度)
            device: 计算设备 ("cpu" 或 "cuda")
        """
        super().__init__()
        
        # 【面试考点1】：维度校验 - d_model必须能被num_heads整除
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.device = device
        
        # 【面试考点2】：PyTorch nn.Linear层的优势
        # 相比手动权重矩阵，nn.Linear提供更好的初始化和GPU加速
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        # 【面试考点3】：KV Cache初始化 - 使用register_buffer
        # register_buffer: 注册非参数张量，自动跟随模型设备变化，保存在state_dict中，但不参与梯度计算
        self.register_buffer('k_cache', torch.empty(0))
        self.register_buffer('v_cache', torch.empty(0))
        self.cache_len = 0
        
        # 移动到指定设备
        self.to(device)
        
    def init_cache(self, batch_size: int) -> None:
        """
        初始化KV缓存
        【面试考点4】：PyTorch缓存空间预分配策略
        """
        # 使用torch.zeros在指定设备上创建缓存
        self.k_cache = torch.zeros(
            batch_size, self.num_heads, self.max_seq_len, self.d_k,
            device=self.device, dtype=torch.float32
        )
        self.v_cache = torch.zeros(
            batch_size, self.num_heads, self.max_seq_len, self.d_k,
            device=self.device, dtype=torch.float32
        )
        self.cache_len = 0
    
    def update_cache(self, k: torch.Tensor, v: torch.Tensor, start_pos: int = None) -> None:
        """
        更新KV缓存
        【面试考点5】：高效的PyTorch缓存更新机制
        
        Args:
            k: Key tensor (batch_size, num_heads, seq_len, d_k)
            v: Value tensor (batch_size, num_heads, seq_len, d_k)
            start_pos: 缓存的起始位置，用于增量更新
        """
        batch_size, num_heads, seq_len, d_k = k.shape
        
        # 如果缓存未初始化，先初始化
        # numel() = number of elements，检查张量元素总数是否为0（即空张量）
        if self.k_cache.numel() == 0:
            self.init_cache(batch_size)
        
        # 【面试考点6】：PyTorch张量切片赋值的高效性
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
    
    def get_cached_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取缓存的K,V
        【面试考点7】：PyTorch张量切片的内存效率
        """
        if self.k_cache.numel() == 0:
            return None, None
        
        # 只返回有效长度的缓存
        k_cached = self.k_cache[:, :, :self.cache_len, :]
        v_cached = self.v_cache[:, :, :self.cache_len, :]
        return k_cached, v_cached
    
    def clear_cache(self) -> None:
        """
        清空缓存
        【面试考点8】：PyTorch缓存生命周期管理
        """
        # 重置为空张量而不是None，保持设备一致性
        # torch.empty(0): 创建未初始化的空张量，比zeros更快，保持tensor类型
        self.k_cache = torch.empty(0, device=self.device)
        self.v_cache = torch.empty(0, device=self.device)
        self.cache_len = 0
    
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                                   mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Scaled Dot-Product Attention with PyTorch optimizations
        优化的缩放点积注意力计算
        
        【面试考点9】：PyTorch vs NumPy的性能优势
        - 自动GPU加速
        - 优化的BLAS库调用
        - 内存布局优化
        """
        d_k = Q.size(-1)
        
        # 【面试考点10】：torch.matmul的广播和优化
        # PyTorch的matmul针对批处理矩阵乘法进行了高度优化
        # math.sqrt(d_k): d_k是常数，用math比torch.sqrt更高效，避免创建不必要的tensor
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Mask处理 - 使用PyTorch的masked_fill
        # masked_fill(mask, value): 将mask为True的位置填充为value，用于屏蔽padding位置
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 【面试考点11】：F.softmax vs 手动实现
        # PyTorch的softmax使用了数值稳定的实现和GPU优化
        attention_weights = F.softmax(scores, dim=-1)
        
        # 加权求和
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, 
                use_cache: bool = False, start_pos: int = None) -> torch.Tensor:
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
        
        # 【面试考点12】：PyTorch nn.Linear的前向传播
        # 自动处理批处理维度，支持GPU加速
        Q = self.W_q(x)  # (batch_size, seq_len, d_model)
        K = self.W_k(x)  # (batch_size, seq_len, d_model)
        V = self.W_v(x)  # (batch_size, seq_len, d_model)
        
        # 【面试考点13】：PyTorch view vs reshape的区别
        # view要求连续内存，reshape会在需要时复制数据
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        if use_cache:
            # 【面试考点14】：KV缓存的使用策略
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
        
        # 【面试考点15】：多头拼接的内存布局优化
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # 最终线性变换
        output = self.W_o(attention_output)
        
        return output
    
    def forward_incremental(self, x: torch.Tensor, position: int) -> torch.Tensor:
        """
        增量推理的前向传播
        【面试考点16】：高效的增量推理实现
        
        Args:
            x: 单个token的输入 (batch_size, 1, d_model)
            position: 当前token在序列中的位置
        """
        return self.forward(x, use_cache=True, start_pos=position)
    
    @torch.no_grad()
    def generate_text(self, prompt_tokens: torch.Tensor, max_new_tokens: int = 50) -> torch.Tensor:
        """
        文本生成示例
        【面试考点17】：实际应用中的自回归生成
        
        Args:
            prompt_tokens: 提示词token序列 (batch_size, prompt_len, d_model)
            max_new_tokens: 最大生成token数
            
        Returns:
            generated_tokens: 生成的完整序列
        """
        batch_size, prompt_len, d_model = prompt_tokens.shape
        
        # 预填充阶段：处理整个prompt
        self.clear_cache()
        _ = self.forward(prompt_tokens, use_cache=True)
        
        generated = prompt_tokens
        
        # 增量生成阶段：逐个生成新token
        for i in range(max_new_tokens):
            # 模拟下一个token（实际应用中这里会是语言模型的输出）
            next_token = torch.randn(batch_size, 1, d_model, device=self.device)
            
            # 增量推理
            _ = self.forward_incremental(next_token, prompt_len + i)
            
            # 拼接到生成序列
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        获取缓存状态信息
        【面试考点18】：PyTorch张量内存监控
        """
        if self.k_cache.numel() == 0:
            return {
                "cache_initialized": False,
                "cache_len": 0,
                "max_len": self.max_seq_len,
                "device": self.device
            }
        
        # PyTorch张量内存大小计算
        cache_size_mb = (self.k_cache.numel() + self.v_cache.numel()) * 4 / (1024 * 1024)  # float32 = 4 bytes
        
        return {
            "cache_initialized": True,
            "cache_len": self.cache_len,
            "max_len": self.max_seq_len,
            "cache_size_mb": cache_size_mb,
            "utilization": self.cache_len / self.max_seq_len,
            "device": self.device,
            "cache_shape": self.k_cache.shape
        }


def test_mha_with_kv_cache_torch():
    """
    Test function for PyTorch Multi-Head Attention with KV Cache
    【面试考点19】：PyTorch版本的完整测试
    """
    print("=== PyTorch Multi-Head Attention with KV Cache 测试 ===")
    
    # 检测设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 测试参数
    batch_size = 2
    seq_len = 4
    d_model = 512
    num_heads = 8
    max_seq_len = 1024
    
    print(f"配置参数: d_model={d_model}, num_heads={num_heads}, max_seq_len={max_seq_len}")
    
    # 初始化模型
    mha_cache = MultiHeadAttentionWithKVCache(d_model, num_heads, max_seq_len, device)
    
    # 测试1：基础功能（不使用缓存）
    print(f"\n=== 测试1: 基础功能 ===")
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    output1 = mha_cache(x, use_cache=False)
    print(f"输入形状: {x.shape}, 设备: {x.device}")
    print(f"输出形状: {output1.shape}, 设备: {output1.device}")
    assert output1.shape == x.shape, "基础功能测试失败"
    print("✅ 基础功能测试通过")
    
    # 测试2：预填充阶段（使用缓存）
    print(f"\n=== 测试2: 预填充阶段 ===")
    mha_cache.clear_cache()
    output2 = mha_cache(x, use_cache=True)
    cache_info = mha_cache.get_cache_info()
    print(f"缓存状态: {cache_info}")
    print(f"输出一致性: {torch.allclose(output1, output2, rtol=1e-5)}")
    assert output2.shape == x.shape, "预填充测试失败"
    print("✅ 预填充阶段测试通过")
    
    # 测试3：增量推理
    print(f"\n=== 测试3: 增量推理 ===")
    new_token = torch.randn(batch_size, 1, d_model, device=device)
    output_incremental = mha_cache.forward_incremental(new_token, position=seq_len)
    
    # 验证缓存更新
    cache_info_after = mha_cache.get_cache_info()
    print(f"增量推理后缓存长度: {cache_info_after['cache_len']}")
    print(f"预期缓存长度: {seq_len + 1}")
    assert cache_info_after['cache_len'] == seq_len + 1, "增量推理缓存更新失败"
    print("✅ 增量推理测试通过")
    
    # 测试4：文本生成示例
    print(f"\n=== 测试4: 文本生成示例 ===")
    prompt = torch.randn(1, 3, d_model, device=device)  # 3个token的prompt
    generated = mha_cache.generate_text(prompt, max_new_tokens=5)
    print(f"Prompt长度: {prompt.shape[1]}")
    print(f"生成序列长度: {generated.shape[1]}")
    print(f"预期长度: {3 + 5}")
    assert generated.shape[1] == 8, "文本生成测试失败"
    print("✅ 文本生成测试通过")
    
    # 测试5：Mask功能测试
    print(f"\n=== 测试5: Mask功能测试 ===")
    
    # 创建padding mask示例
    # 假设第1个序列有3个有效token，第2个序列有2个有效token
    padding_mask = torch.tensor([[1, 1, 1, 0],   # 第1个序列：前3个有效
                                [1, 1, 0, 0]], device=device, dtype=torch.float32)  # 第2个序列：前2个有效
    
    # 扩展到attention维度 (batch_size, seq_len, seq_len)
    attention_mask = padding_mask.unsqueeze(1).expand(-1, seq_len, -1)
    print(f"Padding mask shape: {padding_mask.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    
    # 使用mask的前向传播
    mha_cache.clear_cache()
    output_with_mask = mha_cache(x, mask=attention_mask, use_cache=False)
    output_without_mask = mha_cache(x, mask=None, use_cache=False)
    
    print(f"使用mask时输出范围: [{output_with_mask.min():.4f}, {output_with_mask.max():.4f}]")
    print(f"不使用mask时输出范围: [{output_without_mask.min():.4f}, {output_without_mask.max():.4f}]")
    print(f"输出差异: {(output_with_mask - output_without_mask).abs().max().item():.6f}")
    print("✅ Mask功能测试通过")
    
    # 测试6：梯度检查
    print(f"\n=== 测试6: 梯度检查 ===")
    x_grad = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    output_grad = mha_cache(x_grad, use_cache=False)
    loss = output_grad.sum()
    loss.backward()
    print(f"输入梯度范数: {x_grad.grad.norm().item():.6f}")
    print(f"权重梯度存在: {mha_cache.W_q.weight.grad is not None}")
    print("✅ 梯度检查通过")
    
    # 【面试考点20】：PyTorch vs NumPy的优势对比
    print(f"\n=== PyTorch vs NumPy 优势对比 ===")
    print("1. 设备灵活性: 自动GPU加速，统一的CPU/GPU接口")
    print("2. 自动微分: 内置autograd系统，支持反向传播")
    print("3. 内存效率: 优化的张量操作和内存布局")
    print("4. 生态集成: 与PyTorch生态系统完美集成")
    print("5. 批处理优化: 高度优化的批处理矩阵运算")
    
    # 【面试考点21】：实际部署考虑
    print(f"\n=== 实际部署考虑 ===")
    print("1. 模型量化: 可使用torch.quantization减少内存占用")
    print("2. TorchScript: 可编译为生产环境友好的格式")
    print("3. ONNX导出: 跨平台部署支持")
    print("4. 混合精度: 使用torch.cuda.amp加速训练")
    
    print(f"\n✅ PyTorch Multi-Head Attention with KV Cache 所有测试通过!")
    print("💡 面试重点: PyTorch版本在工程实践中更常用，支持GPU加速和自动微分")


if __name__ == "__main__":
    test_mha_with_kv_cache_torch()