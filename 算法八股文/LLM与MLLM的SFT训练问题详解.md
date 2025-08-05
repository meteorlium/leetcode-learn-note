# LLM与MLLM的SFT训练问题详解

## 核心要点
- **LLM SFT训练**：数据质量、过拟合、灾难性遗忘是三大核心问题
- **MLLM SFT训练**：模态对齐、视觉特征退化、计算资源消耗是关键挑战
- **解决策略**：数据工程、训练策略、模型架构优化三管齐下

## 一、LLM SFT训练常见问题

### 1. 数据质量问题

**问题表现**：
- 训练数据存在噪声、偏见或格式不一致
- 指令-回答对质量参差不齐
- 数据分布不均衡

**解决方法**：
```python
# 数据质量检查示例
def check_data_quality(dataset: List[Dict[str, str]]) -> Dict[str, float]:
    """检查SFT训练数据质量指标"""
    quality_metrics = {
        'avg_instruction_length': 0,
        'avg_response_length': 0,
        'empty_responses': 0,
        'duplicate_pairs': 0
    }
    
    instructions = [item['instruction'] for item in dataset]
    responses = [item['response'] for item in dataset]
    
    # 计算平均长度
    quality_metrics['avg_instruction_length'] = sum(len(inst.split()) for inst in instructions) / len(instructions)
    quality_metrics['avg_response_length'] = sum(len(resp.split()) for resp in responses) / len(responses)
    
    # 检查空回答
    quality_metrics['empty_responses'] = sum(1 for resp in responses if len(resp.strip()) == 0) / len(responses)
    
    return quality_metrics
```

**面试考点**：如何评估和提升SFT数据质量？

### 2. 过拟合问题

**问题表现**：
- 模型在训练集上表现良好，但泛化能力差
- 对特定指令格式过度敏感
- 输出多样性下降

**解决策略**：
- **Early Stopping**：监控验证集loss变化
- **Dropout增强**：在微调时适当增加dropout rate
- **数据增强**：指令重写、同义词替换
- **正则化**：L2正则化、梯度裁剪

```python
# Early Stopping实现
class EarlyStopping:
    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0
        
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
        return self.wait >= self.patience
```

### 3. 灾难性遗忘

**问题表现**：
- 原有能力大幅下降
- 预训练知识被覆盖
- 模型退化到简单模式匹配

**解决方法**：
- **LoRA微调**：只训练低秩分解矩阵
- **混合数据训练**：SFT数据 + 预训练数据
- **知识蒸馏**：保持与原模型的一致性
- **渐进式微调**：分阶段调整学习率

### 4. 学习率设置问题

**关键原则**：
- SFT学习率通常比预训练小1-2个数量级
- 采用warmup策略避免初期震荡
- 使用cosine decay或linear decay

```python
# SFT训练学习率调度
def get_sft_scheduler(optimizer, num_training_steps: int, warmup_ratio: float = 0.1):
    """SFT训练的学习率调度器"""
    from transformers import get_cosine_schedule_with_warmup
    
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
```

## 二、MLLM SFT训练特有问题

### 1. 模态对齐问题

**问题表现**：
- 视觉特征与文本特征语义空间不匹配  
- 视觉信息在文本生成中权重不足
- 跨模态理解能力退化

**BLIP2解决方案**：
- **Q-Former架构**：可学习查询向量连接视觉和语言
- **两阶段训练**：先视觉-语言对齐，再指令微调

```python
# BLIP2风格的模态对齐
class QFormerAlignment:
    def __init__(self, vision_dim: int, text_dim: int, num_queries: int = 32):
        self.query_tokens = nn.Parameter(torch.randn(num_queries, text_dim))
        self.cross_attention = nn.MultiheadAttention(text_dim, num_heads=8)
        
    def forward(self, vision_features, text_features):
        # 使用可学习查询向量对齐视觉特征
        aligned_features, _ = self.cross_attention(
            self.query_tokens.unsqueeze(0).repeat(vision_features.size(0), 1, 1),
            vision_features,
            vision_features
        )
        return aligned_features
```

**LLaVA解决方案**：
- **线性投影层**：简单高效的特征对齐
- **预训练对齐**：大规模图像-文本对预训练

### 2. 视觉特征退化

**问题表现**：
- SFT过程中视觉编码器性能下降
- 细粒度视觉信息丢失
- 视觉推理能力减弱

**解决策略**：
- **冻结视觉编码器**：只训练连接层和语言模型
- **多尺度特征**：使用不同层级的视觉特征
- **视觉增强损失**：额外的视觉重建损失

```python
# 视觉特征保护策略
class VisualFeatureProtection:
    def __init__(self, vision_encoder, freeze_layers: int = -1):
        self.vision_encoder = vision_encoder
        
        # 冻结指定层数
        if freeze_layers > 0:
            for i, layer in enumerate(vision_encoder.layers):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
    
    def extract_multiscale_features(self, images):
        """提取多尺度视觉特征"""
        features = []
        x = images
        
        for i, layer in enumerate(self.vision_encoder.layers):
            x = layer(x)
            if i in [6, 12, 18, 24]:  # 选择特定层
                features.append(x)
        
        return features
```

### 3. 计算资源消耗

**问题表现**：
- 显存占用巨大（视觉编码器 + 大语言模型）
- 训练时间长
- 推理速度慢

**优化策略**：

| 优化方法 | 显存节省 | 性能影响 | 适用场景 |
|---------|---------|---------|---------|
| 梯度检查点 | 40-60% | 训练速度降低20% | 大模型训练 |
| LoRA微调 | 70-80% | 性能基本无损 | 资源受限 |
| 混合精度 | 30-50% | 可能轻微性能下降 | 通用场景 |
| 模型并行 | 按GPU数平分 | 通信开销 | 多GPU环境 |

```python
# MLLM内存优化配置
from transformers import TrainingArguments

def get_mllm_training_args(output_dir: str):
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,  # 小批次大小
        gradient_accumulation_steps=32,  # 梯度累积
        gradient_checkpointing=True,    # 梯度检查点
        fp16=True,                      # 混合精度
        dataloader_pin_memory=False,    # 减少内存占用
        remove_unused_columns=False,    # 保留所有列
    )
```

### 4. 指令理解偏差

**问题表现**：
- 对视觉相关指令理解不准确
- 忽略图像信息直接生成文本
- 视觉推理链不完整

**解决方法**：
- **指令增强**：明确视觉相关的指令格式
- **Chain-of-Thought**：引导模型进行视觉推理
- **多轮对话**：渐进式视觉理解

## 三、实践经验与技巧

### 1. 训练策略

**阶段性训练**：
1. **预对齐阶段**：大量图像-描述对训练
2. **指令微调阶段**：多样化指令数据训练  
3. **强化学习阶段**：基于人类反馈优化

**数据配比**：
- 纯文本指令：40%
- 单图指令：50%  
- 多图指令：10%

### 2. 评估指标

**自动评估**：
- BLEU/ROUGE：文本生成质量
- CIDEr：图像描述准确性
- VQA准确率：视觉问答性能

**人工评估**：
- 指令遵循度
- 视觉理解准确性
- 回答有用性

### 3. 调试技巧

```python
# MLLM训练监控
def monitor_mllm_training(model, dataloader):
    """监控MLLM训练过程"""
    vision_grad_norm = 0
    text_grad_norm = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            if 'vision' in name:
                vision_grad_norm += param.grad.norm().item() ** 2
            else:
                text_grad_norm += param.grad.norm().item() ** 2
    
    print(f"Vision grad norm: {vision_grad_norm ** 0.5:.4f}")
    print(f"Text grad norm: {text_grad_norm ** 0.5:.4f}")
```

## 四、面试常考问题

1. **为什么MLLM比纯文本LLM更难训练？**
   - 模态对齐困难、计算复杂度高、数据质量要求更严格

2. **如何判断SFT训练是否过拟合？**
   - 验证集指标恶化、输出多样性下降、泛化能力测试

3. **BLIP2和LLaVA的主要区别？**
   - BLIP2用Q-Former对齐，LLaVA用线性投影；训练策略和架构复杂度不同

4. **SFT训练中学习率如何设置？**
   - 比预训练小1-2个数量级，使用warmup和decay策略

这份文档涵盖了LLM和MLLM SFT训练的核心问题和解决方案，重点突出实用性和面试相关性。