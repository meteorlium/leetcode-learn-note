import numpy as np

def demonstrate_axis_parameter():
    """
    演示 axis 参数的作用 - 深度学习中的重要概念
    """
    print("=== axis 参数详解 ===\n")
    
    # 创建一个 3x4 的二维数组作为示例
    x = np.array([
        [1, 8, 3, 2],    # 第0行
        [5, 2, 7, 1],    # 第1行  
        [9, 4, 6, 3]     # 第2行
    ])
    
    print("原始数组 x:")
    print(x)
    print(f"形状: {x.shape}")
    print()
    
    # axis=None (默认) - 计算整个数组的最大值
    max_all = np.max(x)
    print(f"axis=None (全局最大值): {max_all}")
    print(f"结果形状: {np.array([max_all]).shape}")
    print()
    
    # axis=0 - 沿着第0维(行方向)计算，即每一列的最大值
    max_axis0 = np.max(x, axis=0)
    print(f"axis=0 (每列最大值): {max_axis0}")
    print(f"结果形状: {max_axis0.shape}")
    print("解释: [max(1,5,9), max(8,2,4), max(3,7,6), max(2,1,3)]")
    print()
    
    # axis=1 - 沿着第1维(列方向)计算，即每一行的最大值  
    max_axis1 = np.max(x, axis=1)
    print(f"axis=1 (每行最大值): {max_axis1}")
    print(f"结果形状: {max_axis1.shape}")
    print("解释: [max(1,8,3,2), max(5,2,7,1), max(9,4,6,3)]")
    print()
    
    # keepdims=True 的作用
    print("=== keepdims 参数的作用 ===")
    
    max_axis1_keepdims = np.max(x, axis=1, keepdims=True)
    print(f"axis=1, keepdims=True: {max_axis1_keepdims}")
    print(f"结果形状: {max_axis1_keepdims.shape}")
    print("注意：保持了原始维度，便于广播运算")
    print()
    
    # 在 Softmax 中的实际应用
    print("=== 在 Softmax 中的应用 ===")
    
    # 模拟 batch_size=3, num_classes=4 的logits
    logits = x.astype(float)
    print("输入 logits:")
    print(logits)
    print()
    
    # 计算每个样本的最大值 (axis=-1 等价于 axis=1)
    max_vals = np.max(logits, axis=-1, keepdims=True)
    print("每行最大值 (axis=-1, keepdims=True):")
    print(max_vals)
    print()
    
    # 减去最大值确保数值稳定
    logits_stable = logits - max_vals
    print("减去最大值后 (数值稳定):")
    print(logits_stable)
    print()
    
    # 计算 softmax
    exp_vals = np.exp(logits_stable)
    sum_exp = np.sum(exp_vals, axis=-1, keepdims=True)
    softmax = exp_vals / sum_exp
    
    print("Softmax 结果:")
    print(softmax)
    print(f"每行和: {np.sum(softmax, axis=1)}")


def demonstrate_3d_axis():
    """
    演示三维数组中的 axis 参数
    """
    print("\n=== 三维数组中的 axis 参数 ===")
    
    # 创建形状为 (2, 3, 4) 的三维数组
    # 可以理解为：2个样本，每个样本3个序列位置，每个位置4维特征
    x_3d = np.random.randint(1, 10, size=(2, 3, 4))
    
    print("三维数组:")
    print(x_3d)
    print(f"形状: {x_3d.shape}")
    print()
    
    # axis=0: 沿第0维计算 (样本维度)
    max_axis0 = np.max(x_3d, axis=0)
    print(f"axis=0 结果形状: {max_axis0.shape}")
    print("含义: 每个位置上，所有样本的最大值")
    print()
    
    # axis=1: 沿第1维计算 (序列维度)  
    max_axis1 = np.max(x_3d, axis=1)
    print(f"axis=1 结果形状: {max_axis1.shape}")
    print("含义: 每个样本中，所有序列位置的最大值")
    print()
    
    # axis=2: 沿第2维计算 (特征维度)
    max_axis2 = np.max(x_3d, axis=2)
    print(f"axis=2 结果形状: {max_axis2.shape}")
    print("含义: 每个位置上，所有特征的最大值")
    print()
    
    # axis=-1: 等价于 axis=2 (最后一维)
    max_axis_neg1 = np.max(x_3d, axis=-1)
    print(f"axis=-1 结果形状: {max_axis_neg1.shape}")
    print(f"与 axis=2 相同: {np.array_equal(max_axis2, max_axis_neg1)}")


def softmax_axis_examples():
    """
    Softmax 中不同 axis 的实际意义
    """
    print("\n=== Softmax 中 axis 的实际意义 ===")
    
    # 场景1: 批处理的分类问题
    # 形状: (batch_size=2, num_classes=3)
    logits_classification = np.array([
        [2.0, 1.0, 0.5],  # 样本1的类别得分
        [1.5, 0.8, 2.2]   # 样本2的类别得分
    ])
    
    print("分类任务 logits (batch_size=2, num_classes=3):")
    print(logits_classification)
    
    # axis=-1: 对每个样本的类别维度计算 softmax
    softmax_classification = softmax_func(logits_classification, axis=-1)
    print("\nSoftmax 结果 (axis=-1, 每行是一个概率分布):")
    print(softmax_classification)
    print(f"每行和: {np.sum(softmax_classification, axis=1)}")
    print()
    
    # 场景2: 序列到序列任务
    # 形状: (batch_size=1, seq_len=3, vocab_size=4)
    logits_seq2seq = np.random.randn(1, 3, 4)
    
    print("序列生成任务 logits (batch_size=1, seq_len=3, vocab_size=4):")
    print(logits_seq2seq[0])  # 只显示第一个样本
    
    # axis=-1: 对词汇表维度计算 softmax
    softmax_seq2seq = softmax_func(logits_seq2seq, axis=-1)
    print("\nSoftmax 结果 (axis=-1, 每个位置的词汇概率分布):")
    print(softmax_seq2seq[0])
    print(f"每个位置的概率和: {np.sum(softmax_seq2seq[0], axis=1)}")


def softmax_func(x, axis=-1):
    """简单的 softmax 实现"""
    x_stable = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_stable)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


if __name__ == "__main__":
    demonstrate_axis_parameter()
    demonstrate_3d_axis()
    softmax_axis_examples()
    
    print("\n=== 总结 ===")
    print("axis 参数的核心作用:")
    print("- axis=0: 沿第0维(通常是batch维度)计算")
    print("- axis=1: 沿第1维(通常是序列或特征维度)计算") 
    print("- axis=-1: 沿最后一维(通常是类别或词汇维度)计算")
    print("- keepdims=True: 保持原维度，便于广播运算")
    print("\n在深度学习中，通常:")
    print("- 分类任务: axis=-1 (对类别维度做softmax)")
    print("- 注意力机制: axis=-1 (对序列长度做softmax)")
    print("- 数值稳定: 总是先减去该维度的最大值")