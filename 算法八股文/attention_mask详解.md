# Attention Mask 详解

## 概述
Attention Mask 是注意力机制中的重要组件，用于控制模型在计算注意力时应该"关注"哪些位置，忽略哪些位置。

## 核心概念

### 1. Mask 的基本原理
```python
# 注意力分数计算
scores = Q @ K.T / sqrt(d_k)

# 应用mask：将无效位置设为很大的负数
scores = scores.masked_fill(mask == 0, -1e9)

# Softmax后，-1e9的位置概率接近0
attention_weights = softmax(scores)  # 无效位置权重≈0
```

## 常见的 Mask 类型

### 1. Padding Mask（填充掩码）
**用途**：屏蔽序列中的padding token
```python
# 原始序列（不同长度）
sequences = [
    ["hello", "world", "!"],           # 长度3
    ["I", "love", "AI", "research"],   # 长度4
    ["yes"]                            # 长度1
]

# Padding后的序列
padded_sequences = [
    ["hello", "world", "!", "<pad>"],
    ["I", "love", "AI", "research"], 
    ["yes", "<pad>", "<pad>", "<pad>"]
]

# Padding mask (1=有效，0=padding)
padding_mask = torch.tensor([
    [1, 1, 1, 0],  # 前3个有效
    [1, 1, 1, 1],  # 全部有效
    [1, 0, 0, 0]   # 只有第1个有效
])
```

### 2. Causal Mask（因果掩码）
**用途**：自回归模型中防止"看到未来"
```python
# 生成下三角矩阵
seq_len = 4
causal_mask = torch.tril(torch.ones(seq_len, seq_len))
print(causal_mask)
# tensor([[1., 0., 0., 0.],    # 位置0只能看到自己
#         [1., 1., 0., 0.],    # 位置1能看到0,1
#         [1., 1., 1., 0.],    # 位置2能看到0,1,2
#         [1., 1., 1., 1.]])   # 位置3能看到0,1,2,3

# 在GPT等模型中的应用
# "I love AI" -> 预测下一个词时
# - "I" 只能基于 "I"
# - "love" 只能基于 "I love" 
# - "AI" 只能基于 "I love AI"
```

### 3. 组合 Mask
**用途**：同时考虑padding和因果关系
```python
def create_combined_mask(input_ids, pad_token_id=0):
    batch_size, seq_len = input_ids.shape
    
    # 1. 创建padding mask
    pad_mask = (input_ids != pad_token_id).float()
    # shape: (batch_size, seq_len)
    
    # 2. 创建causal mask  
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    # shape: (seq_len, seq_len)
    
    # 3. 组合：广播相乘
    combined_mask = pad_mask.unsqueeze(1) * causal_mask.unsqueeze(0)
    # shape: (batch_size, seq_len, seq_len)
    
    return combined_mask

# 示例
input_ids = torch.tensor([
    [1, 2, 3, 0],  # 序列1：前3个有效
    [4, 5, 0, 0]   # 序列2：前2个有效  
])

# 运行结果：
# Input IDs:
# tensor([[1, 2, 3, 0],
#         [4, 5, 0, 0]])

# Padding mask: (只标记有效位置)
# tensor([[1., 1., 1., 0.],  # 序列1：前3个有效
#         [1., 1., 0., 0.]]) # 序列2：前2个有效

# Causal mask: (下三角矩阵)
# tensor([[1., 0., 0., 0.],  # 位置0只能看到位置0
#         [1., 1., 0., 0.],  # 位置1能看到位置0,1
#         [1., 1., 1., 0.],  # 位置2能看到位置0,1,2
#         [1., 1., 1., 1.]]) # 位置3能看到位置0,1,2,3

# Combined mask: (同时满足padding和causal约束)
# tensor([[[1., 0., 0., 0.],  # 序列1
#          [1., 1., 0., 0.],  # - 位置0,1,2有效
#          [1., 1., 1., 0.],  # - 位置3是padding，全部屏蔽
#          [1., 1., 1., 0.]], # - 注意最后一行：即使causal允许看到位置3，但padding约束屏蔽了
#
#         [[1., 0., 0., 0.],  # 序列2  
#          [1., 1., 0., 0.],  # - 位置0,1有效
#          [1., 1., 0., 0.],  # - 位置2,3是padding，全部屏蔽
#          [1., 1., 0., 0.]]])# - 每行的后两列都是0（padding屏蔽）

mask = create_combined_mask(input_ids, pad_token_id=0)
```

## 面试考点

### 考点1：Mask的数学原理
```python
# Q: 为什么用-1e9而不是0？
# A: softmax的特性
scores = torch.tensor([1.0, 2.0, -1e9, 3.0])
probs = F.softmax(scores, dim=-1)
print(probs)  # tensor([0.0900, 0.2447, 0.0000, 0.6652])
# -1e9经过softmax后接近0，有效屏蔽该位置
```

### 考点2：Mask的维度处理
```python
# Q: Mask的shape如何处理？
# A: 需要匹配attention矩阵的维度

# 输入: (batch_size, seq_len, d_model)
# Q,K,V: (batch_size, num_heads, seq_len, d_k)  
# scores: (batch_size, num_heads, seq_len, seq_len)
# mask需要: (batch_size, seq_len, seq_len) 或可广播的shape

# 常见的mask扩展方式
padding_mask = torch.tensor([[1, 1, 0, 0]])  # (1, 4)
attention_mask = padding_mask.unsqueeze(1).expand(-1, 4, -1)  # (1, 4, 4)
```

### 考点3：不同框架的实现
```python
# PyTorch
scores = scores.masked_fill(mask == 0, -1e9)

# TensorFlow  
scores = tf.where(mask == 0, -1e9, scores)

# 手动实现
scores = scores + (1 - mask) * (-1e9)
```

## 实际应用场景

### 1. BERT（双向编码器）
```python
# 只使用padding mask，允许双向attention
def bert_attention_mask(input_ids, pad_token_id=0):
    # 创建padding mask
    mask = (input_ids != pad_token_id).float()
    # 扩展到attention维度
    attention_mask = mask.unsqueeze(1).unsqueeze(2)
    return attention_mask
```

### 2. GPT（自回归生成）
```python  
# 同时使用padding mask和causal mask
def gpt_attention_mask(input_ids, pad_token_id=0):
    batch_size, seq_len = input_ids.shape
    
    # Padding mask
    pad_mask = (input_ids != pad_token_id).float()
    
    # Causal mask
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    
    # 组合
    attention_mask = pad_mask.unsqueeze(1) * causal_mask.unsqueeze(0)
    return attention_mask
```

### 3. 机器翻译
```python
# Encoder用padding mask，Decoder用causal+padding mask
def translation_masks(src_ids, tgt_ids, pad_token_id=0):
    # Encoder self-attention: 只需padding mask
    src_mask = (src_ids != pad_token_id).float()
    
    # Decoder self-attention: causal + padding
    tgt_mask = create_combined_mask(tgt_ids, pad_token_id)
    
    # Cross-attention: src的padding mask
    cross_mask = src_mask.unsqueeze(1).expand(-1, tgt_ids.size(1), -1)
    
    return src_mask, tgt_mask, cross_mask
```

## 性能优化技巧

### 1. 预计算Mask
```python
# 避免每次forward都创建causal mask
class CachedCausalMask:
    def __init__(self, max_seq_len):
        self.max_seq_len = max_seq_len
        self.causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
    
    def get_mask(self, seq_len):
        return self.causal_mask[:seq_len, :seq_len]
```

### 2. 内存优化
```python
# 使用view而不是expand节省内存
def efficient_mask_expansion(mask, target_shape):
    # mask: (batch_size, seq_len)
    # target: (batch_size, seq_len, seq_len)
    return mask.unsqueeze(-1) * mask.unsqueeze(-2)
```

## 常见错误

### 1. Mask值搞反
```python
# ❌ 错误：0表示attend，1表示mask
scores = scores.masked_fill(mask == 1, -1e9)

# ✅ 正确：1表示attend，0表示mask  
scores = scores.masked_fill(mask == 0, -1e9)
```

### 2. 维度不匹配
```python
# ❌ 错误：mask维度不匹配
mask = torch.ones(batch_size, seq_len)  # 缺少一个维度
scores = scores.masked_fill(mask == 0, -1e9)  # 报错

# ✅ 正确：确保维度匹配
mask = mask.unsqueeze(1).expand(-1, seq_len, -1)
```

### 3. 设备不一致
```python
# ❌ 错误：mask在CPU，scores在GPU
mask = torch.ones(2, 4, 4)  # CPU
scores = scores.masked_fill(mask == 0, -1e9)  # 报错

# ✅ 正确：确保设备一致
mask = mask.to(scores.device)
```

## 总结

Attention Mask 是控制注意力机制的关键工具：
- **Padding Mask**: 屏蔽无效token
- **Causal Mask**: 防止看到未来信息  
- **组合使用**: 适应不同的模型需求
- **性能优化**: 预计算和内存管理
- **常见陷阱**: 值的含义、维度匹配、设备一致性