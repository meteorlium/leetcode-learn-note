# BatchNormalization (批标准化)

## 算法原理

### 核心思想
BatchNormalization 通过标准化每一层的输入来解决深度神经网络训练中的内部协变量偏移(Internal Covariate Shift)问题，使训练更稳定、收敛更快。

### 数学公式
对于一个 mini-batch 的输入 x = {x₁, x₂, ..., xₘ}：

1. **计算均值和方差**：
   - μ = (1/m) Σᵢ₌₁ᵐ xᵢ
   - σ² = (1/m) Σᵢ₌₁ᵐ (xᵢ - μ)²

2. **标准化**：
   - x̂ᵢ = (xᵢ - μ) / √(σ² + ε)
   - ε 是防止除零的小常数

3. **缩放和平移**：
   - yᵢ = γx̂ᵢ + β
   - γ 和 β 是可学习参数

### 算法步骤
```python
def batch_norm(x, gamma, beta, eps=1e-5):
    # x: [N, C, H, W] 或 [N, C]
    # 计算 batch 维度的统计量
    mu = x.mean(dim=0, keepdim=True)
    var = x.var(dim=0, keepdim=True, unbiased=False)
    
    # 标准化
    x_norm = (x - mu) / torch.sqrt(var + eps)
    
    # 缩放和平移
    out = gamma * x_norm + beta
    return out
```

## 核心优势

### 1. 解决梯度问题
- **梯度消失**：标准化使得梯度在各层间更稳定传播
- **梯度爆炸**：限制了激活值的范围，防止梯度过大

### 2. 加速收敛
- 允许使用更大的学习率
- 减少对权重初始化的敏感性
- 训练速度提升 2-3 倍

### 3. 正则化效果
- 每个 mini-batch 引入噪声，类似 Dropout
- 减少过拟合，提高泛化能力

## 训练与推理差异

### 训练时
- 使用当前 mini-batch 的统计量
- 同时维护全局的移动平均统计量

### 推理时
- 使用训练时累积的移动平均统计量
- 确保推理结果的一致性

```python
# 训练时的移动平均更新
running_mean = momentum * running_mean + (1 - momentum) * batch_mean
running_var = momentum * running_var + (1 - momentum) * batch_var
```

## 面试高频考点

### 1. 为什么需要 BatchNorm？
- **内部协变量偏移**：随着网络加深，层间输入分布发生变化
- **训练不稳定**：梯度消失/爆炸问题
- **收敛缓慢**：需要小心调节学习率和权重初始化

### 2. γ 和 β 参数的作用
- **恒等映射**：当 γ=1, β=0 时，可以还原原始输入
- **学习能力**：网络可以学习到最优的缩放和偏移
- **表达能力**：保证 BN 不会损失网络的表达能力

### 3. BN 的位置选择
```python
# 方案1：Conv -> BN -> ReLU (推荐)
x = conv(x)
x = batch_norm(x)
x = relu(x)

# 方案2：Conv -> ReLU -> BN
x = conv(x)
x = relu(x)
x = batch_norm(x)
```

### 4. 与其他 Normalization 的对比

| 方法 | 标准化维度 | 适用场景 | 优缺点 |
|------|------------|----------|--------|
| BatchNorm | Batch维度 | 大batch训练 | 依赖batch size |
| LayerNorm | 特征维度 | RNN、Transformer | 不依赖batch |
| InstanceNorm | 实例维度 | 风格迁移 | 适合图像任务 |
| GroupNorm | 分组维度 | 小batch训练 | 平衡性能和稳定性 |

## 实现细节与注意事项

### 1. 数值稳定性
```python
# 错误：可能导致数值不稳定
std = torch.sqrt(var)
x_norm = (x - mean) / std

# 正确：添加 epsilon
x_norm = (x - mean) / torch.sqrt(var + eps)
```

### 2. 维度处理
```python
# 2D: [N, C] -> 在 N 维度标准化
# 4D: [N, C, H, W] -> 在 N, H, W 维度标准化，保持 C 维度独立
```

### 3. 反向传播
BN 的梯度计算相对复杂，涉及：
- 对输入 x 的梯度
- 对参数 γ, β 的梯度
- 对均值和方差的梯度传播

## 变种与改进

### 1. Synchronized BatchNorm
- 多GPU训练时同步统计量
- 解决小batch问题

### 2. Batch Renormalization
- 训练时逐渐从batch统计量过渡到移动平均
- 缓解推理时的差异

### 3. Adaptive BatchNorm
- 根据输入动态调整标准化强度
- 适应不同的数据分布

## 实际应用经验

### 1. 何时使用 BN
- ✅ 深度卷积网络 (ResNet, DenseNet)
- ✅ 大batch size训练
- ❌ RNN (使用LayerNorm)
- ❌ batch size=1 的推理

### 2. 调试技巧
- 监控各层的激活值分布
- 检查 γ, β 参数的更新
- 对比有无BN的训练曲线

### 3. 性能优化
- 使用 fused 版本提升速度
- 推理时可以将BN融合到卷积层
- 量化时需要特殊处理BN参数

## 总结

BatchNormalization 是深度学习中的重要技术，通过标准化中间层输入解决了训练稳定性问题，已成为现代深度网络的标准组件。理解其原理、实现和应用场景对深度学习工程师至关重要。