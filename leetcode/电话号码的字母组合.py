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

        def backtrack(pos: int):  # pos: 0,...,len(digits)
            if pos == len(digits):
                # 结束
                combinations.append(''.join(combination))
                return
            # 添加当前pos
            for ch in phone_map[digits[pos]]:
                combination.append(ch)
                backtrack(pos + 1)
                combination.pop()
            return
        

        combination = []  # 当前组合
        combinations = []  # 所有组合
        backtrack(0)
        return combinations

if __name__ == "__main__":
    s = Solution()
    print(s.letterCombinations("23"))
    print(s.letterCombinations(""))
    print(s.letterCombinations("2"))