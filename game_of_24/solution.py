from __future__ import annotations

from typing import List, Optional, Any, Callable, Tuple, Set
from dataclasses import dataclass, field
from collections import deque
from fractions import Fraction


# ---------------------------
# 1) 通用 ToT 框架
# ---------------------------

@dataclass
class ThoughtNode:
    """思维树节点"""
    state: Any
    parent: Optional["ThoughtNode"] = None
    action: Optional[str] = None     # 记录从 parent -> 当前 的操作描述（可选）
    score: float = 0.0               # evaluate(state)
    depth: int = 0                   # 深度（步数）
    # 你也可以加 children，但 BFS 不需要存全树，省内存


class TreeOfThoughts:
    """Tree of Thoughts 实现（先做 BFS 版本）"""

    def __init__(
        self,
        thought_generator: Callable[[Any], List[Tuple[Any, str]]],
        state_evaluator: Callable[[Any], float],
        goal_checker: Callable[[Any], bool],
        state_key: Callable[[Any], Any],
        max_depth: int = 3,
        prune_threshold: float = 0.0,
    ):
        """
        Args:
            thought_generator: 输入 state，返回 [(next_state, action_str), ...]
            state_evaluator: evaluate(state) -> float
            goal_checker: goal(state) -> bool
            state_key: 用于去重的 key(state) -> hashable
            max_depth: 最大搜索深度（24点=3步）
            prune_threshold: 剪枝阈值，低于则不扩展
        """
        self.thought_generator = thought_generator
        self.state_evaluator = state_evaluator
        self.goal_checker = goal_checker
        self.state_key = state_key
        self.max_depth = max_depth
        self.prune_threshold = prune_threshold

    def search(self, initial_state: Any, strategy: str = "bfs") -> Optional[ThoughtNode]:
        """执行搜索：支持 BFS / DFS"""

        root = ThoughtNode(
            state=initial_state,
            parent=None,
            action=None,
            score=self.state_evaluator(initial_state),
            depth=0
        )

        visited: Set[Any] = set()
        visited.add(self.state_key(initial_state))

        if strategy.lower() == "bfs":
            frontier = deque([root])
            pop_node = frontier.popleft
            push_node = frontier.append
        elif strategy.lower() == "dfs":
            frontier = [root]          # 用 list 做栈
            pop_node = frontier.pop    # 后进先出
            push_node = frontier.append
        else:
            raise ValueError("strategy must be 'bfs' or 'dfs'")

        while frontier:
            node = pop_node()

            if self.goal_checker(node.state):
                return node

            if node.depth >= self.max_depth:
                continue

            if node.score < self.prune_threshold:
                continue

            for next_state, action_str in self.thought_generator(node.state):
                k = self.state_key(next_state)
                if k in visited:
                    continue
                visited.add(k)

                child = ThoughtNode(
                    state=next_state,
                    parent=node,
                    action=action_str,
                    score=self.state_evaluator(next_state),
                    depth=node.depth + 1
                )
                push_node(child)

        return None


# ---------------------------
# 2) 24点求解器（把问题塞进 ToT 框架）
# ---------------------------

# state 采用 (nums, exprs)
# nums: List[Fraction]
# exprs: List[str] 与 nums 同步


class Point24Solver:
    """24点求解器"""

    def __init__(self):
        # 目标值
        self.target = Fraction(24, 1)

        # 创建 ToT：注入 problem-specific 的 generator/eval/goal/key
        self.tot = TreeOfThoughts(
            thought_generator=self._expand_state,
            state_evaluator=self._evaluate_state,
            goal_checker=self._is_goal,
            state_key=self._state_key,
            max_depth=3,                 # 4个数 -> 1个数：正好3步
            prune_threshold=0.0,         # 先不激进剪枝，保证正确性
        )

    def solve(self, numbers: List[int], strategy: str = "bfs") -> Optional[str]:
        """
        求解24点
        Args:
            numbers: 4个数字 (1-13)
        Returns:
            表达式字符串，无解返回None
        """
        if len(numbers) != 4:
            raise ValueError("numbers must have length 4")

        nums = [Fraction(x, 1) for x in numbers]
        exprs = [str(x) for x in numbers]
        initial_state = (nums, exprs)

        # goal_node = self.tot.search(initial_state)
        goal_node = self.tot.search(initial_state, strategy=strategy)
        if goal_node is None:
            return None

        # goal_node.state 只剩一个表达式
        final_nums, final_exprs = goal_node.state
        if len(final_exprs) != 1:
            return None
        return final_exprs[0]

    # --------- problem-specific methods ---------

    def _is_goal(self, state: Any) -> bool:
        nums, _ = state
        return len(nums) == 1 and nums[0] == self.target

    def _evaluate_state(self, state: Any) -> float:
        """
        evaluate(state) -> [0,1]
        简单版本：看离24最近的那个数
        """
        nums, _ = state
        # 距离越小越好
        d = min(abs(x - self.target) for x in nums)
        score_dist = float(1 / (1 + d))  # d=0 => 1.0

        # 分母复杂度惩罚（分母越大越差）
        max_den = max(x.denominator for x in nums)
        score_frac = 1.0 / (1.0 + max_den)  # den=1 => 0.5（偏保守，但可解释）

        # 合成：距离为主，复杂度为辅
        score = 0.85 * score_dist + 0.15 * score_frac

        # 裁剪到[0,1]
        if score < 0.0:
            return 0.0
        if score > 1.0:
            return 1.0
        return score

    def _state_key(self, state: Any) -> Tuple[Tuple[int, int], ...]:
        """
        用 nums 的多重集合做去重 key。
        只用数值去重（不看表达式），能大幅减少重复搜索。
        """
        nums, _ = state
        pairs = [(x.numerator, x.denominator) for x in nums]
        pairs.sort()
        return tuple(pairs)

    def _expand_state(self, state: Any) -> List[Tuple[Any, str]]:
        """
        从当前 state 生成下一步候选。
        返回: [(next_state, action_str), ...]
        """
        nums, exprs = state
        n = len(nums)
        results: List[Tuple[Any, str]] = []

        # 选两张牌 i<j
        for i in range(n):
            for j in range(i + 1, n):
                a, b = nums[i], nums[j]
                ea, eb = exprs[i], exprs[j]

                # 剩余部分
                rest_nums = [nums[k] for k in range(n) if k != i and k != j]
                rest_exprs = [exprs[k] for k in range(n) if k != i and k != j]

                # 生成候选操作
                # + 和 * 只做一次（避免交换律重复）
                candidates: List[Tuple[Fraction, str]] = []
                candidates.append((a + b, f"({ea}+{eb})"))
                candidates.append((a * b, f"({ea}*{eb})"))

                # - 两个方向
                candidates.append((a - b, f"({ea}-{eb})"))
                candidates.append((b - a, f"({eb}-{ea})"))

                # / 两个方向，注意除0
                if b != 0:
                    candidates.append((a / b, f"({ea}/{eb})"))
                if a != 0:
                    candidates.append((b / a, f"({eb}/{ea})"))

                # 组合成 next_state
                for val, e in candidates:
                    next_nums = rest_nums + [val]
                    next_exprs = rest_exprs + [e]
                    results.append(((next_nums, next_exprs), e))

        return results


# ---------------------------
# 3) 简单测试
# ---------------------------
if __name__ == "__main__":
    solver = Point24Solver()
    
    print(solver.solve([3,3,8,8], strategy="bfs"))  # 可能输出类似: (8/(3-(8/3)))
    
    print(solver.solve([3,3,8,8], strategy="dfs"))
    
