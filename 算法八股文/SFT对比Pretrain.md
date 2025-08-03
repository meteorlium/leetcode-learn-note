# SFT对比Pretrain深度解析

## 面试核心考点
**主要动机差异**: Pretrain学习通用语言知识，SFT学习指令遵循能力

## 一、训练目标与动机

| 维度 | Pretrain | SFT |
|------|----------|-----|
| **核心目标** | 学习语言的统计规律和世界知识 | 学习如何理解和遵循人类指令 |
| **输出特征** | 续写文本，可能不符合用户意图 | 直接回答问题，符合用户期望 |
| **应用场景** | 基础语言模型 | 对话助手、指令执行 |

## 二、数据构成差异

### Pretrain数据特点
- **规模**: 万亿级token（如CommonCrawl、Wikipedia）
- **质量**: 质量参差不齐，包含大量噪声
- **来源**: 互联网爬取、书籍、新闻等
- **格式**: 连续文本，无特定结构

### SFT数据特点  
- **规模**: 通常几万到几十万条对话
- **质量**: 人工标注，质量极高
- **来源**: 人工构造的指令-回答对
- **格式**: 结构化的`<instruction, response>`对

## 三、训练技术核心差异

### 损失函数计算
```python
# Pretrain: 全序列损失
loss = cross_entropy(logits, labels)  # 所有token

# SFT: 仅response部分损失  
loss = cross_entropy(logits[response_mask], labels[response_mask])
```

### 数据处理流程
```python
# Pretrain格式
text = "今天天气很好，适合出门散步..."

# SFT格式
formatted = f"<|user|>{instruction}<|assistant|>{response}<|end|>"
# 只对response部分计算loss
```

## 四、训练策略对比

| 维度 | Pretrain | SFT |
|------|----------|-----|
| **学习率** | 1e-4 ~ 5e-4 | 1e-5 ~ 5e-5 |
| **训练轮数** | 1-2 epoch | 3-5 epoch |
| **batch size** | 较大(2M-4M tokens) | 较小(32-128样本) |
| **梯度累积** | 常用 | 必需 |

## 五、技术实现细节

### Attention Mask差异
- **Pretrain**: 标准causal mask，防止看到未来token
- **SFT**: 特殊mask，instruction部分不参与loss计算

### 模型初始化
- **Pretrain**: 随机初始化或继续训练
- **SFT**: 基于pretrain模型checkpoint开始

## 六、面试常见问题

### Q1: 为什么SFT不对instruction计算loss？
**答**: instruction是输入，模型应该学习基于instruction生成response，而不是预测instruction本身。

### Q2: SFT会导致catastrophic forgetting吗？
**答**: 会，因此需要：
- 较小学习率
- 混合pretrain数据
- 或使用LoRA等参数高效微调

### Q3: SFT数据量相比pretrain很小，为什么有效？
**答**: SFT是task-specific微调，少量高质量数据就能显著改变模型行为模式。

## 七、实际应用考虑

### 数据质量控制
- 人工review确保response质量
- 多样性检查避免模型过拟合
- 安全性过滤避免有害内容

### 训练监控指标
- Training loss收敛情况
- 验证集上的instruction following能力  
- 人工评估response质量

## 核心要点总结
1. **本质相同**: 都是next token prediction
2. **目标不同**: 语言建模 vs 指令遵循
3. **数据差异**: 大规模原始文本 vs 小规模结构化对话
4. **技术细节**: 全序列loss vs 部分loss，不同训练策略