# src/trajectory.py
import numpy as np
from src.utils import info, success

def simulate_trajectory(env, policy, start_state_idx):
    """
    依据给定策略从起点状态模拟完整轨迹
    
    参数:
        env: InertialRacingEnv 实例
        policy: numpy 数组，shape (num_states,)，每个元素为动作索引 (0~8)
        start_state_idx: 起始状态的索引
    
    返回:
        dict: {
            'path': [(x0,y0), (x1,y1), ..., (xn,yn)],  # 坐标序列（含起点和终点）
            'total_reward': float,                      # 累计总回报
            'steps': int,                               # 步数
            'success': bool                             # 是否成功到达终点
        }
        
    注意:
        - 使用方案A：实时累计环境奖励，不依赖任何V值
        - 每步步惩罚为 -10，到达终点时额外获得 goal_reward
        - 如果遇到死亡状态，标记为失败并返回已走路径
    """
    path = []
    total_reward = 0.0
    current_idx = start_state_idx
    
    # 获取起始位置的坐标
    cur, prev = env.get_state_info(current_idx)
    path.append(cur)  # 添加起点坐标
    
    max_steps = 1000  # 防止无限循环的安全限制
    steps = 0
    success = False
    
    while steps < max_steps:
        # 获取当前状态应采取的动作
        action_idx = policy[current_idx]
        
        # 查询转移结果
        typ, data = env.get_transition(current_idx, action_idx)
        
        if typ == 'death':
            # 撞墙或出界，轨迹终止
            total_reward += -np.inf
            break
        
        elif typ == 'goal':
            # 到达终点
            reward = -10.0 + data  # 步惩罚 + 终点奖励
            total_reward += reward
            
            # 获取终点坐标（需要从当前状态和动作推断）
            cur, prev = env.get_state_info(current_idx)
            vx = cur[0] - prev[0]
            vy = cur[1] - prev[1]
            temp = (cur[0] + vx, cur[1] + vy)
            goal_coord = (temp[0] + env.actions[action_idx][0], 
                         temp[1] + env.actions[action_idx][1])
            path.append(goal_coord)  # 添加终点坐标
            
            success = True
            steps += 1
            break
        
        else:  # typ == 'normal'
            # 正常移动
            total_reward += -10.0  # 步惩罚
            next_idx = data
            
            # 获取下一个位置的坐标
            next_cur, _ = env.get_state_info(next_idx)
            path.append(next_cur)
            
            # 转移到下一状态
            current_idx = next_idx
            steps += 1
    
    return {
        'path': path,
        'total_reward': total_reward,
        'steps': steps,
        'success': success
    }


def find_best_start_and_path(env, policy, return_all=False):
    """
    遍历所有起点，模拟轨迹，找到总回报最大的路径
    
    参数:
        env: InertialRacingEnv 实例
        policy: numpy 数组，shape (num_states,)，策略
        return_all: bool，是否返回所有起点的结果列表
    
    返回:
        如果 return_all=False:
            dict 或 None: 最优路径的字典，如果没有可达起点则返回 None
        
        如果 return_all=True:
            tuple: (best_result, all_results_list)
                - best_result: 最优路径的字典（或 None）
                - all_results_list: 所有起点的结果列表（包含成功和失败路径）
    
    注意:
        - 保留所有路径（包括失败路径 success=False）
        - 排序规则：成功路径优先于失败路径
        - 成功路径按 total_reward 从高到低排序
        - 失败路径排在最后
    """
    # 步骤1：获取所有起点状态索引
    start_indices = env.get_start_state_indices()
    
    if not start_indices:
        raise ValueError("环境中没有可达的起点状态")
    
    info(f"初始格子数: {len(start_indices)}")
    
    # 步骤2：遍历所有起点，模拟轨迹（保留所有结果）
    all_results = []
    for i, start_idx in enumerate(start_indices):
        # info(f"正在处理第 {i+1}/{len(start_indices)} 个起点...")
        result = simulate_trajectory(env, policy, start_idx)
        
        # 添加起点信息到结果中
        cur, prev = env.get_state_info(start_idx)
        result['start_idx'] = start_idx
        result['start_coord'] = cur
        all_results.append(result)
    
    # 步骤3：排序 - 成功路径优先，然后按回报降序
    def sort_key(result):
        if result['success']:
            return (1, result['total_reward'])  # 成功路径优先
        else:
            return (0, -np.inf)  # 失败路径排最后
    
    all_results.sort(key=sort_key, reverse=True)
    
    # 步骤4：处理没有路径的情况（理论上不会发生）
    if not all_results:
        if return_all:
            return None, []
        else:
            return None
    
    # 步骤5：找出最优路径（第一个元素）
    best_result = all_results[0]
    
    # 输出成功信息
    if best_result['success']:
        success(f"✓ 找到最优路径! 总回报: {best_result['total_reward']:.2f}, 步数: {best_result['steps']}")
    else:
        from src.utils import warning
        warning(f"⚠ 所有路径均失败,最优路径总回报: {best_result['total_reward']:.2f}")
    
    # 步骤6：根据 return_all 决定返回值
    if return_all:
        return best_result, all_results
    else:
        return best_result


def analyze_policy_slices_paths(env, policy_slices):
    """
    对策略切片中的每个策略，计算其最优起点路径
    
    参数:
        env: InertialRacingEnv 实例
        policy_slices: list of numpy arrays，策略切片列表
                      （通常来自 sample_history_slices 的 policy_slices 输出）
    
    返回:
        list of dict: 每个元素对应一个策略切片的最优路径结果
                     （即 find_best_start_and_path 返回的 best_result）
    
    示例:
        >>> from src.history_utils import sample_history_slices
        >>> V_slices, policy_slices, indices = sample_history_slices(V_history, policy_history)
        >>> best_paths = analyze_policy_slices_paths(env, policy_slices)
        >>> print(f"分析了 {len(best_paths)} 个策略切片")
        >>> print(f"第0轮最优路径代价: {-best_paths[0]['total_reward']}")
    
    注意:
        - 与 sample_history_slices 配合使用，分析迭代过程中的策略演化
        - 每个策略切片独立计算最优路径
        - 仅返回最优路径（best_result），不包含所有起点的详细结果
    """
    best_results = []
    
    for i, policy in enumerate(policy_slices):
        info(f"正在处理策略切片 {i+1}/{len(policy_slices)}...")
        best_result = find_best_start_and_path(env, policy, return_all=False)
        best_results.append(best_result)
    
    return best_results
