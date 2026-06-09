# src/history_utils.py
"""
迭代历史分析工具模块

提供从价值迭代历史记录中采样、分析等功能。
"""
import numpy as np
from src.utils import info


def sample_history_slices(V_history, policy_history, num_slices=6):
    """
    从迭代历史中均匀采样指定数量的切片
    
    参数:
        V_history: list of numpy arrays，价值函数历史记录
                   每个元素为某一轮迭代结束后的 V 数组副本
        policy_history: list of numpy arrays，策略历史记录
                        每个元素为对应轮次的贪婪策略动作数组
        num_slices: 采样数量（默认6）
    
    返回:
        tuple: (V_slices, policy_slices, iteration_indices)
            - V_slices: list of numpy arrays，采样的价值函数
            - policy_slices: list of numpy arrays，采样的策略
            - iteration_indices: list of int，对应的迭代轮数编号
              （第0个元素对应初始状态，尚未开始迭代）
    
    注意:
        - 始终包含第0轮（初始状态）和最后一轮（收敛状态）
        - 中间均匀分布 num_slices-2 个切片
        - 如果历史长度 < num_slices，返回所有历史
        - 使用 np.linspace 确保均匀分布
    
    示例:
        >>> V_slices, policy_slices, indices = sample_history_slices(
        ...     V_history, policy_history, num_slices=6)
        >>> print(f"采样了 {len(indices)} 个切片")
        >>> print(f"迭代轮数: {indices}")
        # 输出: 采样了 6 个切片
        #       迭代轮数: [0, 20, 40, 60, 80, 100]
    """
    total_iterations = len(V_history)
    
    info(f"总迭代数: {total_iterations}")
    
    if total_iterations <= num_slices:
        # 历史长度不足，返回全部
        indices = list(range(total_iterations))
    else:
        # 均匀采样：包含首尾
        # np.linspace 生成 num_slices 个均匀分布的点
        indices = np.linspace(0, total_iterations - 1, num_slices, dtype=int)
        # 去重并保持顺序（防止因浮点误差导致重复）
        indices = sorted(set(indices.tolist()))
    
    info(f"采样切片编号: {indices}")
    
    # 提取对应的 V 和 policy 切片
    V_slices = [V_history[i] for i in indices]
    policy_slices = [policy_history[i] for i in indices]
    
    return V_slices, policy_slices, indices
