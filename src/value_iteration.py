# src/value_iteration.py
import numpy as np
from src.utils import success, warning, info
import time

def value_iteration(env, gamma=1.0, theta=1e-6, max_iterations=10000, record_history=False, return_stats=False):
    """
    异步价值迭代（Gauss-Seidel 就地更新）
    
    参数:
        env: InertialRacingEnv 实例
        gamma: 折扣因子，固定为1.0（情节式任务）
        theta: 收敛阈值，价值变化小于该值时停止
        max_iterations: 最大迭代轮数
        record_history: 是否记录每轮迭代后的价值数组和策略快照
        return_stats: 是否返回统计信息
    
    返回:
        V: 最优价值函数，numpy数组 shape (num_states,)
        policy: 最优策略，numpy数组 shape (num_states,) 每个元素是动作索引
        V_history: 如果 record_history=True，返回每轮迭代后的 V 副本列表；否则 None
        policy_history: 如果 record_history=True，返回每轮迭代后的贪婪策略列表；否则 None

        根据参数组合返回不同内容：
        - 若 return_stats=False 且 record_history=False: (V, policy)
        - 若 return_stats=False 且 record_history=True:  (V, policy, V_history, policy_history)
        - 若 return_stats=True 且 record_history=False: (V, policy, stats)
        - 若 return_stats=True 且 record_history=True:  (V, policy, V_history, policy_history, stats)
    """
    num_states = env.get_num_states()
    num_actions = env.get_num_actions()
    
    V = np.zeros(num_states, dtype=float)
    
    V_history = []
    policy_history = []
    
    if record_history:
        # 记录初始策略（基于初始 V 全零）
        init_policy = np.zeros(num_states, dtype=int)
        V_history.append(V.copy())
        policy_history.append(init_policy.copy())
    
    # 记录开始时间
    start_time = time.time()

    for iteration in range(max_iterations):
        delta = 0.0
        for s in range(num_states):
            old_value = V[s]
            best_q = -np.inf
            for a in range(num_actions):
                typ, data = env.get_transition(s, a)
                if typ == 'death':
                    q = -np.inf
                elif typ == 'goal':
                    q = -10.0 + data
                else:
                    q = -10.0 + gamma * V[data]
                if q > best_q:
                    best_q = q
            V[s] = best_q
            delta = max(delta, abs(old_value - V[s]))
        
        # 记录本轮迭代后的贪婪策略
        if record_history:
            policy_iter = np.zeros(num_states, dtype=int)
            for s in range(num_states):
                best_q = -np.inf
                best_a = 0
                for a in range(num_actions):
                    typ, data = env.get_transition(s, a)
                    if typ == 'death':
                        q = -np.inf
                    elif typ == 'goal':
                        q = -10.0 + data
                    else:
                        q = -10.0 + gamma * V[data]
                    if q > best_q:
                        best_q = q
                        best_a = a
                policy_iter[s] = best_a
            policy_history.append(policy_iter.copy())
            V_history.append(V.copy())
        
        if delta < theta:
            success(f"价值迭代收敛! 迭代轮数: {iteration+1}, delta={delta:.6f}")
            break
    else:
        warning(f"价值迭代达到最大迭代次数 ({max_iterations}) 仍未收敛")

    # 最终最优策略
    policy = np.zeros(num_states, dtype=int)
    for s in range(num_states):
        best_q = -np.inf
        best_a = 0
        for a in range(num_actions):
            typ, data = env.get_transition(s, a)
            if typ == 'death':
                q = -np.inf
            elif typ == 'goal':
                q = -10.0 + data
            else:
                q = -10.0 + gamma * V[data]
            if q > best_q:
                best_q = q
                best_a = a
        policy[s] = best_a
    
    elapsed_time = time.time() - start_time
    # 转成毫秒
    elapsed_time_ms = elapsed_time * 1000

    stats = {
        'converged': delta < theta,
        'iterations': iteration + 1,
        'final_delta': delta,
        'time_ms': elapsed_time_ms
    }

    # 打印统计信息
    info(f"是否收敛: {stats['converged']}, 迭代轮数: {stats['iterations']}, 最终 delta: {stats['final_delta']:.6f}, 耗时: {stats['time_ms']:.2f} 毫秒")

    # 根据参数返回不同内容
    if record_history:
        if return_stats:
            return V, policy, V_history, policy_history, stats
        else:
            return V, policy, V_history, policy_history
    else:
        if return_stats:
            return V, policy, stats
        else:
            return V, policy