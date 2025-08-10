"""
leetcode 17: 电话号码的字母组合
https://leetcode.cn/problems/letter-combinations-of-a-phone-number/description/

给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回。

给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。

示例 1：

输入：digits = "23"
输出：["ad","ae","af","bd","be","bf","cd","ce","cf"]
示例 2：

输入：digits = ""
输出：[]
示例 3：

输入：digits = "2"
输出：["a","b","c"]
 

提示：

0 <= digits.length <= 4
digits[i] 是范围 ['2', '9'] 的一个数字。
"""

# 思路：回溯

"""
回溯法写代码关键注意事项（面试重点）：

1. 函数设计原则：
   - 参数设计：通过位置参数（如pos）控制递归深度，避免重复遍历
   - 边界条件：明确递归终止条件，通常是处理完所有位置时收集结果
   
2. 变量设计核心：
   - combination: 当前路径结果（动态变化）
     * 使用list而非str，便于append/pop操作
     * 在递归过程中共享同一个对象，节省空间
   - combinations: 全局结果收集（静态累积）  
     * 收集时需要深拷贝当前路径：''.join(combination)
     * 不能直接append(combination)，会导致引用问题
     
3. 回溯三步骤（经典模板）：
   - 做选择：combination.append(ch)
   - 递归：backtrack(pos + 1) 
   - 撤销选择：combination.pop()
   
4. 面试常见错误：
   - 忘记撤销选择导致状态污染
   - 结果收集时直接引用而非拷贝
   - 递归参数设计不当导致重复计算
"""

from typing import List

class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if digits == '':
            return []
        
        phone_map = {
            "2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "pqrs",
            "8": "tuv",
            "9": "wxyz",
        }

        def backtrack(pos: int):  # pos: 0,...,len(digits) 【面试考点：递归参数设计】
            if pos == len(digits):
                # 【面试考点】边界条件：处理完所有位置
                combinations.append(''.join(combination))  # 【关键】深拷贝当前路径，避免引用问题
                return
            
            # 【面试考点】回溯核心：遍历当前位置所有可能选择
            for ch in phone_map[digits[pos]]:
                combination.append(ch)      # 1. 做选择
                backtrack(pos + 1)          # 2. 递归下一层 
                combination.pop()           # 3. 撤销选择【面试易错点：忘记撤销】
            return
        

        combination = []   # 【设计要点】当前路径：动态变化，共享对象节省空间
        combinations = []  # 【设计要点】结果集合：静态累积，收集所有合法路径
        backtrack(0)
        return combinations

if __name__ == "__main__":
    s = Solution()
    print(s.letterCombinations("23"))
    print(s.letterCombinations(""))
    print(s.letterCombinations("2"))