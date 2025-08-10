"""
leetcode 146: LRU 缓存
https://leetcode.cn/problems/lru-cache/description/

请你设计并实现一个满足 LRU (最近最少使用) 缓存 约束的数据结构。
实现 LRUCache 类：
- LRUCache(int capacity) 以 正整数 作为容量 capacity 初始化 LRU 缓存
- int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1
- void put(int key, int value) 如果关键字 key 已经存在，则变更其数据值 value；如果不存在，则向缓存中插入该组 key-value。如果插入操作导致关键字数量超过 capacity，则应该逐出最久未使用的关键字。

函数 get 和 put 必须以 O(1) 的平均时间复杂度运行。

示例：
输入
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
输出
[null, null, null, 1, null, -1, null, -1, 3, 4]

解释
LRUCache lRUCache = new LRUCache(2);
lRUCache.put(1, 1); // 缓存是 {1=1}
lRUCache.put(2, 2); // 缓存是 {1=1, 2=2}
lRUCache.get(1);    // 返回 1
lRUCache.put(3, 3); // 该操作会使得关键字 2 作废，缓存是 {1=1, 3=3}
lRUCache.get(2);    // 返回 -1 (未找到)
lRUCache.put(4, 4); // 该操作会使得关键字 1 作废，缓存是 {4=4, 3=3}
lRUCache.get(1);    // 返回 -1 (未找到)
lRUCache.get(3);    // 返回 3
lRUCache.get(4);    // 返回 4
"""

# 思路：哈希表 + 双向链表

"""
LRU缓存设计核心要点（面试重点）：

1. 数据结构选择关键：
   - 哈希表：提供O(1)查找，key -> 节点映射
   - 双向链表：提供O(1)插入/删除，维护访问顺序
   - 组合优势：既能快速定位又能快速调整顺序

2. 虚拟头尾节点设计（面试经典技巧）：
   - head: 最近使用端（新访问的放这里）
   - tail: 最久未使用端（要删除的从这里）
   - 好处：统一边界处理，避免空链表判断

3. 关键操作时间复杂度分析：
   - get(key): 哈希查找O(1) + 移动到头部O(1) = O(1)
   - put(key,val): 哈希查找O(1) + 插入/更新O(1) + 可能的删除O(1) = O(1)

4. 面试常见问题：
   - 为什么不用单链表？无法O(1)删除中间节点
   - 为什么不只用哈希表？无法维护访问时间顺序
   - 节点存key的作用？删除尾节点时需要key更新哈希表

5. 面试易错点：
   - 忘记在节点中存储key值
   - 更新现有key时忘记移动位置
   - 删除节点后忘记更新哈希表
"""

from typing import Dict

class Node:
    def __init__(self, key: int = 0, value: int = 0):
        self.key = key      # 【面试要点】存储key用于删除时更新哈希表
        self.value = value
        self.prev = None
        self.next = None


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: Dict[int, Node] = {}  # 【核心】key -> node 映射，O(1)查找
        
        # 【关键设计】虚拟头尾节点，简化边界处理
        self.head = Node()  # head.next 指向最近使用
        self.tail = Node()  # tail.prev 指向最久未使用
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add_to_head(self, node: Node) -> None:
        """在头部添加节点 - O(1)操作"""
        # 【面试考点】双向链表插入操作的标准步骤
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node: Node) -> None:
        """删除指定节点 - O(1)操作"""
        # 【面试考点】双向链表删除操作的标准步骤
        node.prev.next = node.next
        node.next.prev = node.prev

    def _move_to_head(self, node: Node) -> None:
        """移动节点到头部表示最近使用"""
        self._remove_node(node)
        self._add_to_head(node)

    def _pop_tail(self) -> Node:
        """删除尾部节点并返回"""
        last_node = self.tail.prev
        self._remove_node(last_node)
        return last_node

    def get(self, key: int) -> int:
        """获取值并更新为最近使用"""
        node = self.cache.get(key)
        if not node:
            return -1
        
        # 【关键】移动到头部表示最近访问
        self._move_to_head(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        """插入或更新key-value"""
        node = self.cache.get(key)
        
        if not node:
            # 【情况1】新key插入
            new_node = Node(key, value)
            
            if len(self.cache) >= self.capacity:
                # 【LRU核心】超容量时删除最久未使用
                tail_node = self._pop_tail()
                del self.cache[tail_node.key]  # 【易错点】别忘记更新哈希表
            
            # 添加新节点
            self.cache[key] = new_node
            self._add_to_head(new_node)
        else:
            # 【情况2】已存在key的更新
            node.value = value
            self._move_to_head(node)  # 【关键】更新也要标记为最近使用


if __name__ == "__main__":
    # 测试用例：验证LRU缓存功能
    lru = LRUCache(2)
    
    lru.put(1, 1)
    lru.put(2, 2)
    print(f"get(1): {lru.get(1)}")    # 返回 1
    
    lru.put(3, 3)                     # 使key=2失效
    print(f"get(2): {lru.get(2)}")    # 返回 -1
    
    lru.put(4, 4)                     # 使key=1失效  
    print(f"get(1): {lru.get(1)}")    # 返回 -1
    print(f"get(3): {lru.get(3)}")    # 返回 3
    print(f"get(4): {lru.get(4)}")    # 返回 4