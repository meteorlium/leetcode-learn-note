"""
LeetCode 42. 接雨水 (Trapping Rain Water)
https://leetcode.cn/problems/trapping-rain-water/

题目描述：
给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能够接多少雨水。

示例 1：
输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
输出：6
解释：上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。

示例 2：
输入：height = [4,2,0,3,2,5]
输出：9

约束条件：
- n == height.length
- 1 <= n <= 2 * 10^4
- 0 <= height[i] <= 3 * 10^4
"""

from typing import List


class Solution:
    """接雨水问题的多种解法"""
    
    def trap_brute_force(self, height: List[int]) -> int:
        """
        解法1：暴力解法
        
        思路：
        对于每个位置，雨水的高度取决于其左右两边的最高柱子的最小值
        water_level[i] = min(max_left[i], max_right[i]) - height[i]
        
        时间复杂度：O(n²) - 对每个位置都要扫描左右两边
        空间复杂度：O(1)
        """
        if not height or len(height) < 3:
            return 0
        
        n = len(height)
        total_water = 0
        
        # 对每个位置计算能接的雨水
        for i in range(1, n - 1):  # 第一个和最后一个位置不能接雨水
            # 找左边最高的柱子
            max_left = 0
            for j in range(i):
                max_left = max(max_left, height[j])
            
            # 找右边最高的柱子
            max_right = 0
            for j in range(i + 1, n):
                max_right = max(max_right, height[j])
            
            # 计算当前位置能接的雨水
            water_level = min(max_left, max_right)
            if water_level > height[i]:
                total_water += water_level - height[i]
        
        return total_water
    
    def trap_dp(self, height: List[int]) -> int:
        """
        解法2：动态规划
        
        思路：
        预先计算每个位置左边和右边的最大高度，避免重复计算
        
        时间复杂度：O(n) - 三次遍历
        空间复杂度：O(n) - 需要两个数组存储左右最大值
        """
        if not height or len(height) < 3:
            return 0
        
        n = len(height)
        
        # 预计算每个位置左边的最大高度
        max_left = [0] * n
        max_left[0] = height[0]
        for i in range(1, n):
            max_left[i] = max(max_left[i-1], height[i])
        
        # 预计算每个位置右边的最大高度
        max_right = [0] * n
        max_right[n-1] = height[n-1]
        for i in range(n-2, -1, -1):
            max_right[i] = max(max_right[i+1], height[i])
        
        # 计算雨水总量
        total_water = 0
        for i in range(n):
            water_level = min(max_left[i], max_right[i])
            if water_level > height[i]:
                total_water += water_level - height[i]
        
        return total_water
    
    def trap_two_pointers(self, height: List[int]) -> int:
        """
        解法3：双指针
        
        思路：
        使用两个指针从两端向中间移动，维护左右两边的最大高度
        关键洞察：当 left_max < right_max 时，左指针位置的积水只取决于 left_max
        
        时间复杂度：O(n) - 一次遍历
        空间复杂度：O(1) - 只使用常数空间
        """
        if not height or len(height) < 3:
            return 0
        
        left, right = 0, len(height) - 1
        left_max, right_max = 0, 0
        total_water = 0
        
        while left < right:
            if height[left] < height[right]:
                # 左边的柱子更矮，移动左指针
                if height[left] >= left_max:
                    left_max = height[left]
                else:
                    total_water += left_max - height[left]
                left += 1
            else:
                # 右边的柱子更矮，移动右指针
                if height[right] >= right_max:
                    right_max = height[right]
                else:
                    total_water += right_max - height[right]
                right -= 1
        
        return total_water
    
    def trap_stack(self, height: List[int]) -> int:
        """
        解法4：单调栈
        
        思路：
        维护一个单调递减的栈，当遇到更高的柱子时，可以形成凹槽接雨水
        
        时间复杂度：O(n) - 每个元素最多入栈出栈一次
        空间复杂度：O(n) - 栈的最大长度
        """
        if not height or len(height) < 3:
            return 0
        
        stack = []  # 存储柱子的索引
        total_water = 0
        
        for i in range(len(height)):
            # 当前柱子比栈顶柱子高，可以形成凹槽
            while stack and height[i] > height[stack[-1]]:
                bottom = stack.pop()  # 凹槽的底部
                
                if not stack:  # 没有左边界，无法形成凹槽
                    break
                
                # 计算凹槽的宽度和高度
                width = i - stack[-1] - 1
                h = min(height[i], height[stack[-1]]) - height[bottom]
                total_water += width * h
            
            stack.append(i)
        
        return total_water


def test_trapping_rain_water():
    """测试用例"""
    solution = Solution()
    
    test_cases = [
        {
            "input": [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1],
            "expected": 6,
            "description": "示例1：经典接雨水案例"
        },
        {
            "input": [4, 2, 0, 3, 2, 5],
            "expected": 9,
            "description": "示例2：另一个接雨水案例"
        },
        {
            "input": [3, 0, 2, 0, 4],
            "expected": 7,
            "description": "简单案例：两个凹槽"
        },
        {
            "input": [0, 1, 0],
            "expected": 0,
            "description": "边界案例：最小长度无法接雨水"
        },
        {
            "input": [1],
            "expected": 0,
            "description": "边界案例：单个柱子"
        },
        {
            "input": [1, 2, 3, 4, 5],
            "expected": 0,
            "description": "边界案例：递增序列无法接雨水"
        },
        {
            "input": [5, 4, 3, 2, 1],
            "expected": 0,
            "description": "边界案例：递减序列无法接雨水"
        },
        {
            "input": [3, 2, 3],
            "expected": 1,
            "description": "简单案例：一个凹槽"
        }
    ]
    
    methods = [
        ("暴力解法", solution.trap_brute_force),
        ("动态规划", solution.trap_dp),
        ("双指针", solution.trap_two_pointers),
        ("单调栈", solution.trap_stack)
    ]
    
    print("=== 接雨水问题测试 ===\n")
    
    for i, test_case in enumerate(test_cases, 1):
        height = test_case["input"]
        expected = test_case["expected"]
        description = test_case["description"]
        
        print(f"测试用例 {i}: {description}")
        print(f"输入: {height}")
        print(f"期望输出: {expected}")
        
        all_passed = True
        for method_name, method in methods:
            result = method(height.copy())  # 使用副本避免修改原数组
            status = "✓" if result == expected else "✗"
            print(f"  {method_name}: {result} {status}")
            if result != expected:
                all_passed = False
        
        print(f"测试结果: {'✓ 通过' if all_passed else '✗ 失败'}")
        print("-" * 50)


def visualize_rain_water(height: List[int], water_amount: int = None):
    """
    可视化接雨水问题
    """
    if not height:
        return
    
    max_height = max(height)
    n = len(height)
    
    print(f"\n高度数组: {height}")
    if water_amount is not None:
        print(f"接雨水总量: {water_amount}")
    
    # 计算每个位置的水位（使用双指针方法）
    solution = Solution()
    water_levels = [0] * n
    
    # 计算左右最大值
    max_left = [0] * n
    max_right = [0] * n
    
    max_left[0] = height[0]
    for i in range(1, n):
        max_left[i] = max(max_left[i-1], height[i])
    
    max_right[n-1] = height[n-1]
    for i in range(n-2, -1, -1):
        max_right[i] = max(max_right[i+1], height[i])
    
    # 计算每个位置的水位
    for i in range(n):
        water_level = min(max_left[i], max_right[i])
        water_levels[i] = max(0, water_level - height[i])
    
    print("\n可视化图形:")
    print("█ = 柱子, ░ = 雨水, · = 空气")
    
    # 从上到下绘制
    for level in range(max_height, 0, -1):
        line = ""
        for i in range(n):
            if height[i] >= level:
                line += "█"  # 柱子
            elif height[i] + water_levels[i] >= level:
                line += "░"  # 雨水
            else:
                line += "·"  # 空气
        print(f"{level:2d} |{line}|")
    
    # 绘制底部标尺
    print("   " + "+" + "-" * n + "+")
    print("   " + " " + "".join(str(i % 10) for i in range(n)))


if __name__ == "__main__":
    # 运行测试
    test_trapping_rain_water()
    
    # 可视化示例
    print("\n" + "=" * 60)
    print("可视化示例")
    print("=" * 60)
    
    example_height = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
    solution = Solution()
    result = solution.trap_two_pointers(example_height)
    visualize_rain_water(example_height, result)
    
    print("\n")
    example_height2 = [3, 0, 2, 0, 4]
    result2 = solution.trap_two_pointers(example_height2)
    visualize_rain_water(example_height2, result2)