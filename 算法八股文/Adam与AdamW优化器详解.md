# Adam与AdamW优化器详解

## 核心要点
- **Adam**：自适应学习率优化器，结合动量和RMSprop
- **AdamW**：Adam + 权重衰减解耦，LLM训练首选
- **主要优势**：收敛快、自适应、对超参数不敏感
- **关键差异**：权重衰减处理方式不同

## Adam优化器原理

### 算法动机
传统SGD存在的问题：
1. 学习率固定，无法自适应
2. 对所有参数使用相同学习率
3. 收敛速度慢，容易陷入鞍点

Adam解决方案：
- **一阶矩估计**（动量）：加速收敛
- **二阶矩估计**（自适应学习率）：参数级别调整
- **偏差修正**：解决初始化偏差

### 核心算法

```python
import torch
from typing import Dict, List, Optional

class Adam:
    """Adam优化器实现 - 面试重点：理解动量和自适应学习率机制"""
    
    def __init__(self, params: List[torch.Tensor], lr: float = 0.001, 
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas  # 面试考点：为什么是0.9和0.999？
        self.eps = eps
        
        # 初始化状态
        self.m = [torch.zeros_like(p) for p in params]  # 一阶矩
        self.v = [torch.zeros_like(p) for p in params]  # 二阶矩
        self.t = 0  # 时间步
    
    def step(self):
        """面试重点：能手写Adam更新公式"""
        self.t += 1
        
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
                
            grad = param.grad.data
            
            # 更新矩估计
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad.pow(2)
            
            # 偏差修正 - 面试考点：为什么需要偏差修正？
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # 参数更新
            param.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
```

### 数学公式

```
# 梯度：g_t = ∇f(θ_{t-1})
# 动量更新：m_t = β_1 * m_{t-1} + (1-β_1) * g_t
# 自适应项：v_t = β_2 * v_{t-1} + (1-β_2) * g_t²
# 偏差修正：m̂_t = m_t / (1-β_1^t), v̂_t = v_t / (1-β_2^t)  
# 参数更新：θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
```

## AdamW优化器

### 核心改进
AdamW解决Adam的权重衰减问题：
- **问题**：Adam中L2正则化与自适应学习率相互作用
- **解决**：权重衰减与梯度更新解耦

### 关键差异对比

| 方面 | Adam | AdamW |
|------|------|-------|
| 权重衰减 | L2正则化加入梯度 | 直接从参数中减去 |
| 数学表达 | g_t = ∇f(θ) + λθ | θ_t = θ_{t-1} - α(m̂_t/√v̂_t + λθ_{t-1}) |
| 适用场景 | 传统机器学习 | 大模型训练（LLM） |
| 收敛效果 | 权重衰减受学习率影响 | 权重衰减独立控制 |

### AdamW实现

```python
class AdamW:
    """AdamW优化器 - 面试重点：权重衰减解耦机制"""
    
    def __init__(self, params: List[torch.Tensor], lr: float = 0.001,
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.01):  # 典型值0.01-0.1
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay  # 面试考点：为什么解耦？
        
        self.m = [torch.zeros_like(p) for p in params]
        self.v = [torch.zeros_like(p) for p in params]
        self.t = 0
    
    def step(self):
        """面试重点：权重衰减在参数更新阶段执行"""
        self.t += 1
        
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
                
            grad = param.grad.data
            
            # Adam更新（不包含权重衰减）
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad.pow(2)
            
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # 关键：权重衰减解耦执行
            param.data = param.data * (1 - self.lr * self.weight_decay) - \
                        self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
```

## LLM SFT训练应用

### 为什么选择AdamW？

1. **参数量大**：自适应学习率避免手动调优
2. **梯度稀疏**：二阶矩估计处理不均匀更新
3. **正则化需求**：权重衰减防止过拟合
4. **收敛稳定**：偏差修正确保初期稳定性

### 实际配置参数

```python
# LLM SFT训练的典型AdamW配置
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5,              # 面试考点：为什么LLM用小学习率？
    betas=(0.9, 0.999),   # 默认值，适合大多数场景
    eps=1e-8,             # 数值稳定性
    weight_decay=0.01     # 权重衰减系数
)

# 学习率调度器
from transformers import get_cosine_schedule_with_warmup
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,      # 热身步数
    num_training_steps=10000    # 总训练步数
)
```

### 面试常见问题

**Q1: 为什么Adam的β1=0.9, β2=0.999？**
- β1控制动量衰减：0.9平衡历史和当前梯度
- β2控制二阶矩衰减：0.999保留更多历史信息，避免学习率过大

**Q2: AdamW权重衰减为什么要解耦？**
- Adam中L2正则化与自适应学习率相互影响
- 解耦后权重衰减独立控制，效果更好

**Q3: SFT训练为什么用小学习率？**
- 预训练模型已收敛，需要微调而非重新训练
- 小学习率避免破坏预训练知识

**Q4: 如何选择weight_decay？**
- 一般0.01-0.1，模型越大可适当增加
- 通过验证集效果调优

## 与其他优化器对比

| 优化器 | 收敛速度 | 内存开销 | 超参敏感度 | LLM适用性 |
|--------|----------|----------|------------|-----------|
| SGD | 慢 | 低 | 高 | 差 |
| Adam | 快 | 高 | 低 | 一般 |
| AdamW | 快 | 高 | 低 | 优秀 |
| Lion | 快 | 中 | 低 | 优秀 |

## 总结

**面试核心答案**：
1. **Adam**：自适应学习率 + 动量，解决SGD问题
2. **AdamW**：权重衰减解耦，LLM训练更优
3. **关键机制**：一阶矩（动量）+ 二阶矩（自适应）+ 偏差修正
4. **SFT应用**：小学习率 + 权重衰减 + 学习率调度

**记忆要点**：AdamW = Adam + 解耦权重衰减，大模型训练首选优化器。