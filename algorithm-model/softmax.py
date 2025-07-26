import numpy as np
import matplotlib.pyplot as plt


class Softmax:
    def __init__(self):
        """
        Softmax激活函数实现 - 深度学习中的重要激活函数
        
        Softmax的核心作用：
        1. 将任意实数向量转换为概率分布
        2. 输出值在(0,1)之间且和为1
        3. 常用于多分类问题的输出层
        
        数学公式：softmax(xi) = exp(xi) / Σ(exp(xj))
        """
        pass
    
    def forward(self, x, axis=-1):
        """
        Softmax前向传播 - 核心计算函数
        
        Args:
            x: 输入张量 (任意形状)
            axis: 计算softmax的维度，默认为最后一维
            
        Returns:
            输出概率分布
            
        【面试考点1】：数值稳定性处理
        直接计算exp(x)容易数值溢出，减去最大值保证数值稳定
        softmax(x) = softmax(x - max(x)) 数学上等价
        """
        # 【面试考点2】：防止数值溢出的关键步骤
        # 减去最大值不改变softmax结果但避免exp()溢出
        x_stable = x - np.max(x, axis=axis, keepdims=True)
        
        # 【面试考点3】：指数运算
        # exp函数将输入映射到正数域，保证概率非负
        exp_x = np.exp(x_stable)
        
        # 【面试考点4】：归一化步骤
        # 除以总和使得输出为概率分布（和为1）
        softmax_output = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
        
        return softmax_output
    
    def backward(self, y, grad_output):
        """
        Softmax反向传播 - 梯度计算
        
        Args:
            y: softmax的输出 (前向传播结果)
            grad_output: 来自上层的梯度
            
        Returns:
            对输入x的梯度
            
        【面试考点5】：Softmax梯度推导
        ∂softmax(xi)/∂xj = softmax(xi) * (δij - softmax(xj))
        其中δij是Kronecker delta函数
        """
        # 【面试考点6】：雅可比矩阵计算
        # Softmax的梯度是一个雅可比矩阵，不是简单的元素对元素
        batch_size = y.shape[0]
        num_classes = y.shape[1] if len(y.shape) > 1 else len(y)
        
        if len(y.shape) == 1:
            # 一维情况
            grad_input = np.zeros_like(y)
            for i in range(len(y)):
                for j in range(len(y)):
                    if i == j:
                        grad_input[i] += grad_output[j] * y[i] * (1 - y[j])
                    else:
                        grad_input[i] += grad_output[j] * y[i] * (-y[j])
        else:
            # 批处理情况
            grad_input = np.zeros_like(y)
            for b in range(batch_size):
                for i in range(num_classes):
                    for j in range(num_classes):
                        if i == j:
                            grad_input[b, i] += grad_output[b, j] * y[b, i] * (1 - y[b, j])
                        else:
                            grad_input[b, i] += grad_output[b, j] * y[b, i] * (-y[b, j])
        
        return grad_input
    
    def cross_entropy_loss(self, predictions, targets):
        """
        交叉熵损失函数 - Softmax常用的损失函数
        
        Args:
            predictions: softmax输出 (batch_size, num_classes)
            targets: 真实标签 (batch_size, num_classes) one-hot编码
            
        Returns:
            交叉熵损失值
            
        【面试考点7】：为什么Softmax和交叉熵配合使用？
        1. 数学上优雅：组合后梯度形式简单
        2. 数值稳定：可以直接从logits计算避免中间步骤
        3. 概率解释：最大似然估计的自然结果
        """
        # 【面试考点8】：数值稳定的损失计算
        # 添加小常数避免log(0)
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        
        # 【面试考点9】：交叉熵公式 H(p,q) = -Σ p(x)log(q(x))
        loss = -np.sum(targets * np.log(predictions), axis=1)
        return np.mean(loss)
    
    def softmax_with_logits(self, logits, targets):
        """
        数值稳定的Softmax + 交叉熵组合计算
        
        【面试考点10】：工程实现技巧
        直接从logits计算损失，避免中间的softmax步骤，更稳定
        """
        # 数值稳定的实现
        logits_stable = logits - np.max(logits, axis=-1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(logits_stable), axis=-1, keepdims=True))
        log_softmax = logits_stable - log_sum_exp
        
        # 交叉熵损失
        loss = -np.sum(targets * log_softmax, axis=1)
        return np.mean(loss), np.exp(log_softmax)


def demonstrate_softmax_properties():
    """
    演示Softmax的关键性质 - 面试常问问题
    """
    print("=== Softmax算法性质演示 ===")
    
    softmax = Softmax()
    
    # 【面试考点11】：不同输入的Softmax行为
    print("\n1. 基本计算示例:")
    x1 = np.array([1.0, 2.0, 3.0])
    y1 = softmax.forward(x1)
    print(f"输入: {x1}")
    print(f"输出: {y1}")
    print(f"和为1验证: {np.sum(y1):.6f}")
    
    # 【面试考点12】：温度参数的影响
    print("\n2. 温度参数效应:")
    x = np.array([1.0, 2.0, 3.0])
    temperatures = [0.5, 1.0, 2.0, 5.0]
    
    for temp in temperatures:
        y_temp = softmax.forward(x / temp)
        print(f"温度={temp}: {y_temp} (最大值占比: {np.max(y_temp):.3f})")
    
    print("\n【面试重点】温度参数作用:")
    print("- 温度 < 1: 输出更加锐化，接近one-hot")
    print("- 温度 > 1: 输出更加平滑，接近均匀分布")
    print("- 温度 → 0: 输出接近硬性最大值")
    print("- 温度 → ∞: 输出接近均匀分布")
    
    # 【面试考点13】：大数值输入的稳定性
    print("\n3. 数值稳定性验证:")
    x_large = np.array([1000.0, 1001.0, 1002.0])
    try:
        y_naive = np.exp(x_large) / np.sum(np.exp(x_large))
        print(f"直接计算: {y_naive}")
    except:
        print("直接计算溢出!")
    
    y_stable = softmax.forward(x_large)
    print(f"稳定计算: {y_stable}")
    
    return softmax


def test_gradient_computation():
    """
    测试梯度计算 - 反向传播验证
    
    【面试考点14】：数值梯度检验
    使用有限差分法验证解析梯度的正确性
    """
    print("\n=== 梯度计算验证 ===")
    
    softmax = Softmax()
    
    # 简单测试用例
    x = np.array([1.0, 2.0, 3.0])
    y = softmax.forward(x)
    
    # 模拟来自上层的梯度
    grad_output = np.array([0.1, -0.2, 0.1])
    
    # 解析梯度
    grad_analytical = softmax.backward(y, grad_output)
    
    # 数值梯度（有限差分）
    epsilon = 1e-7
    grad_numerical = np.zeros_like(x)
    
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += epsilon
        x_minus[i] -= epsilon
        
        y_plus = softmax.forward(x_plus)
        y_minus = softmax.forward(x_minus)
        
        # 使用链式法则计算数值梯度
        grad_numerical[i] = np.sum(grad_output * (y_plus - y_minus)) / (2 * epsilon)
    
    print(f"解析梯度: {grad_analytical}")
    print(f"数值梯度: {grad_numerical}")
    print(f"差异: {np.abs(grad_analytical - grad_numerical)}")
    
    # 【面试考点15】：梯度检验标准
    tolerance = 1e-5
    is_correct = np.allclose(grad_analytical, grad_numerical, atol=tolerance)
    print(f"梯度验证{'通过' if is_correct else '失败'} (容忍度: {tolerance})")


def classification_example():
    """
    多分类任务示例 - 实际应用场景
    
    【面试考点16】：Softmax在实际问题中的应用
    """
    print("\n=== 多分类任务示例 ===")
    
    softmax = Softmax()
    
    # 模拟3分类问题的logits
    # 假设是图像分类：猫、狗、鸟
    logits = np.array([
        [2.3, 1.1, 0.5],    # 样本1: 更可能是猫
        [0.8, 2.7, 1.2],    # 样本2: 更可能是狗  
        [1.0, 0.9, 2.8],    # 样本3: 更可能是鸟
    ])
    
    # 真实标签 (one-hot编码)
    targets = np.array([
        [1, 0, 0],  # 样本1确实是猫
        [0, 1, 0],  # 样本2确实是狗
        [0, 0, 1],  # 样本3确实是鸟
    ])
    
    # 计算预测概率
    predictions = softmax.forward(logits, axis=1)
    
    print("分类结果:")
    class_names = ['猫', '狗', '鸟']
    for i, (pred, target) in enumerate(zip(predictions, targets)):
        predicted_class = np.argmax(pred)
        true_class = np.argmax(target)
        confidence = pred[predicted_class]
        
        print(f"样本{i+1}: 预测={class_names[predicted_class]}({confidence:.3f}), "
              f"真实={class_names[true_class]}, "
              f"{'✓' if predicted_class == true_class else '✗'}")
    
    # 计算损失
    loss = softmax.cross_entropy_loss(predictions, targets)
    print(f"\n交叉熵损失: {loss:.4f}")
    
    # 【面试考点17】：准确率计算
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(targets, axis=1)
    accuracy = np.mean(predicted_classes == true_classes)
    print(f"准确率: {accuracy:.3f}")


def advanced_concepts():
    """
    高级概念和面试难点
    
    【面试考点18-20】：深入理解
    """
    print("\n=== 高级概念 ===")
    
    print("1. Gumbel-Softmax技巧:")
    print("   - 用于可微分的离散采样")
    print("   - 在强化学习和变分自编码器中应用")
    
    print("\n2. Softmax的替代方案:")
    print("   - Sparsemax: 产生稀疏概率分布")
    print("   - Entmax: 可调节稀疏程度")
    print("   - Hierarchical Softmax: 降低大词汇表的计算复杂度")
    
    print("\n3. 计算优化:")
    print("   - Log-Sum-Exp技巧: 数值稳定的对数空间计算")
    print("   - 向量化实现: 利用SIMD指令加速")
    print("   - 并行化: GPU上的高效实现")
    
    # 【面试考点19】：Softmax vs Sigmoid
    print("\n4. Softmax vs Sigmoid对比:")
    x = np.array([1.0, 2.0])
    softmax = Softmax()
    
    # Softmax (多分类)
    softmax_out = softmax.forward(x)
    
    # Sigmoid (二分类或多标签)
    sigmoid_out = 1 / (1 + np.exp(-x))
    
    print(f"输入: {x}")
    print(f"Softmax输出: {softmax_out} (和={np.sum(softmax_out):.3f})")
    print(f"Sigmoid输出: {sigmoid_out} (独立概率)")
    
    print("\n【关键区别】:")
    print("- Softmax: 概率和为1，用于单标签多分类")
    print("- Sigmoid: 每个输出独立，用于多标签分类")


def test_softmax():
    """
    完整的测试函数 - 算法工程师面试标准
    
    【面试考点21】：全面的测试覆盖
    """
    print("=== Softmax算法深度学习面试测试 ===\n")
    
    # 基本功能测试
    softmax_demo = demonstrate_softmax_properties()
    
    # 梯度验证
    test_gradient_computation()
    
    # 实际应用
    classification_example()
    
    # 高级概念
    advanced_concepts()
    
    print("\n=== 面试总结 ===")
    print("✅ 核心考点覆盖:")
    print("1. 数学原理: 指数归一化，概率分布")
    print("2. 数值稳定性: 减去最大值技巧")
    print("3. 梯度计算: 雅可比矩阵，链式法则")
    print("4. 损失函数: 交叉熵，数值稳定实现")
    print("5. 实际应用: 多分类，温度参数")
    print("6. 工程优化: 向量化，内存效率")
    print("7. 理论对比: vs Sigmoid, vs其他激活函数")
    
    print("\n💡 面试重点回答要点:")
    print("- 能解释Softmax的数学直觉")
    print("- 知道数值稳定性的重要性和实现方法")
    print("- 理解与交叉熵损失的配合使用")
    print("- 掌握梯度推导和反向传播")
    print("- 了解在不同场景下的应用和限制")


if __name__ == "__main__":
    test_softmax()