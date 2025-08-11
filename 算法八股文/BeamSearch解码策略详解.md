# Beam Search解码策略详解

## 核心要点概览
- **动机**：在生成质量和计算效率间平衡，避免greedy search的局部最优
- **本质**：维护k个最优候选序列的宽度优先搜索
- **优势**：比exhaustive search高效，比greedy search质量更好
- **应用**：机器翻译、文本生成、语音识别等序列生成任务

## 算法原理

### 基本思想
Beam Search在每个解码步骤维护k个最优候选序列（beam width = k），通过概率分数选择最有希望的路径继续扩展。

### 算法流程
```python
from typing import List, Tuple
import heapq
import torch

def beam_search(model, input_ids: torch.Tensor, beam_width: int = 5, 
                max_length: int = 50) -> List[Tuple[List[int], float]]:
    """
    Beam Search解码实现
    
    面试考点：
    1. 如何维护beam状态
    2. 概率计算和排序逻辑  
    3. 结束条件判断
    4. 内存和时间复杂度
    """
    batch_size = input_ids.size(0)
    device = input_ids.device
    
    # 初始化：每个样本维护beam_width个候选
    beams = [[(input_ids[i].tolist(), 0.0)] for i in range(batch_size)]
    
    for step in range(max_length):
        all_candidates = []
        
        for batch_idx in range(batch_size):
            candidates = []
            
            for sequence, score in beams[batch_idx]:
                if sequence[-1] == model.eos_token_id:  # 序列已结束
                    candidates.append((sequence, score))
                    continue
                
                # 获取下一个token的概率分布
                input_tensor = torch.tensor([sequence]).to(device)
                with torch.no_grad():
                    outputs = model(input_tensor)
                    logits = outputs.logits[0, -1, :]  # 最后一个位置的logits
                    probs = torch.softmax(logits, dim=-1)
                
                # 获取top-k个候选token
                top_k_probs, top_k_indices = torch.topk(probs, beam_width)
                
                for prob, token_id in zip(top_k_probs, top_k_indices):
                    new_sequence = sequence + [token_id.item()]
                    # 累积对数概率避免数值下溢
                    new_score = score + torch.log(prob).item()
                    candidates.append((new_sequence, new_score))
            
            # 选择top-k个最优候选
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams[batch_idx] = candidates[:beam_width]
    
    return beams
```

## 关键技术点

### 1. 概率计算
```python
# 避免数值下溢的对数概率累积
def calculate_sequence_score(log_probs: List[float], length_penalty: float = 1.0) -> float:
    """
    面试考点：为什么用对数概率？如何处理序列长度偏置？
    """
    # 对数概率求和等于概率连乘
    total_log_prob = sum(log_probs)
    
    # 长度惩罚：避免偏向短序列
    # Google NMT论文公式：score = log_prob / (length^alpha)
    length_penalty_term = len(log_probs) ** length_penalty
    
    return total_log_prob / length_penalty_term
```

### 2. 内存优化
```python
class BeamSearchOptimized:
    """内存优化的Beam Search实现"""
    
    def __init__(self, beam_width: int):
        self.beam_width = beam_width
        self.beam_scores = None
        self.beam_tokens = None
    
    def step(self, logits: torch.Tensor) -> torch.Tensor:
        """
        面试考点：如何减少内存占用？
        - 只保存必要的beam状态
        - 及时释放无用的候选
        - 使用in-place操作
        """
        vocab_size = logits.size(-1)
        
        if self.beam_scores is None:
            # 初始步骤
            log_probs = torch.log_softmax(logits, dim=-1)
            self.beam_scores, indices = torch.topk(log_probs, self.beam_width)
            self.beam_tokens = indices.unsqueeze(1)
        else:
            # 后续步骤
            log_probs = torch.log_softmax(logits, dim=-1)
            # 广播计算所有可能的累积分数
            candidate_scores = self.beam_scores.unsqueeze(1) + log_probs
            candidate_scores = candidate_scores.view(-1)  # 展平
            
            # 选择top-k
            top_scores, top_indices = torch.topk(candidate_scores, self.beam_width)
            
            # 恢复beam和token索引
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size
            
            # 更新状态
            self.beam_scores = top_scores
            self.beam_tokens = torch.cat([
                self.beam_tokens[beam_indices],
                token_indices.unsqueeze(1)
            ], dim=1)
        
        return self.beam_tokens
```

## 主流解码策略对比

| 策略 | 时间复杂度 | 空间复杂度 | 生成质量 | 多样性 | 适用场景 |
|------|------------|------------|----------|--------|----------|
| **Greedy** | O(L×V) | O(1) | 较低 | 极低 | 快速推理 |
| **Beam Search** | O(L×V×K) | O(L×K) | 高 | 低 | 翻译、摘要 |
| **Random Sampling** | O(L×V) | O(1) | 中等 | 高 | 创意写作 |
| **Top-k Sampling** | O(L×V×log k) | O(1) | 中高 | 中高 | 对话生成 |
| **Top-p (Nucleus)** | O(L×V×log V) | O(1) | 中高 | 高 | 多样化生成 |

### 详细对比分析

#### Greedy Search vs Beam Search
```python
# Greedy：每步选择概率最高的token
def greedy_search(logits: torch.Tensor) -> int:
    return torch.argmax(logits, dim=-1).item()

# 问题：容易陷入局部最优
# 例如："The cat sat on the" → "mat"(0.4) vs "couch"(0.3) + "very comfortable"(0.8)
# Greedy选择"mat"，但"couch very comfortable"整体概率更高
```

#### Beam Search vs Sampling方法
```python
def top_k_sampling(logits: torch.Tensor, k: int = 50, temperature: float = 1.0) -> int:
    """
    面试对比要点：
    1. Beam Search确定性 vs Sampling随机性
    2. 质量一致性 vs 多样性
    3. 计算开销差异
    """
    # 温度缩放
    logits = logits / temperature
    
    # Top-k过滤
    top_k_logits, top_k_indices = torch.topk(logits, k)
    
    # 采样
    probs = torch.softmax(top_k_logits, dim=-1)
    sampled_index = torch.multinomial(probs, 1)
    
    return top_k_indices[sampled_index].item()
```

## 实际应用考虑

### 1. 超参数选择
```python
class BeamSearchConfig:
    """
    面试考点：各参数如何影响生成结果？
    """
    def __init__(self):
        self.beam_width = 5          # 平衡质量和效率
        self.length_penalty = 1.0    # 0.6-1.2，避免长度偏置  
        self.repetition_penalty = 1.0 # 1.0-1.5，减少重复
        self.early_stopping = True    # 遇到EOS提前结束
        self.max_length = 512        # 防止无限生成
```

### 2. 工程优化技巧
```python
def beam_search_with_optimizations(
    model, input_ids: torch.Tensor, 
    beam_width: int = 5,
    use_cache: bool = True,  # KV缓存优化
    batch_beam: bool = True  # 批量beam处理
) -> List[str]:
    """
    面试重点：生产环境的优化策略
    1. KV缓存减少重复计算
    2. 批量处理提高GPU利用率
    3. 早停策略节省计算
    4. 动态beam修剪
    """
    if use_cache:
        # 利用transformer的past_key_values缓存
        past_key_values = None
    
    if batch_beam:
        # 将所有beam作为一个批次处理
        # 形状变换：[batch_size, beam_width, seq_len] → [batch_size * beam_width, seq_len]
        pass
    
    # 实现省略...
    return results
```

## 常见面试问题

### Q1: Beam Search相比Greedy的优势在哪里？
**答案要点**：
- Greedy每步局部最优，可能错过全局最优路径
- Beam Search维护多个候选，增加找到更优序列的机会
- 在翻译等任务中，BLEU分数通常有2-3分提升

### Q2: Beam Width如何选择？
**答案要点**：
- 过小(1-2)：接近Greedy，质量提升有限
- 适中(3-10)：质量和效率平衡点
- 过大(>20)：边际收益递减，计算开销大
- 具体选择依赖任务特性和计算资源

### Q3: 如何解决Beam Search的重复问题？
**答案要点**：
1. Repetition Penalty：降低已出现token的概率
2. Coverage Mechanism：跟踪已处理的输入部分
3. Diverse Beam Search：强制不同beam关注不同方面
4. 后处理去重：生成后移除重复的n-gram

### Q4: 为什么用对数概率而不是原始概率？
**答案要点**：
- 避免连乘导致的数值下溢
- 对数运算将乘法转为加法，计算更稳定
- 便于实现长度惩罚等归一化技巧

## 扩展话题

### 1. 变种算法
- **Diverse Beam Search**：增加候选多样性
- **Constrained Beam Search**：满足特定约束条件
- **Stochastic Beam Search**：引入随机性平衡

### 2. 替代方案趋势
- **Contrastive Search**：平衡质量和多样性
- **Typical Sampling**：基于信息论的采样
- **MCTS解码**：蒙特卡洛树搜索用于文本生成

Beam Search作为序列生成的经典算法，在理解其原理基础上，重点掌握与其他策略的对比和实际应用中的优化技巧，这些都是面试的高频考点。