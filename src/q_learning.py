# Q-learning 智能体（无模型，网格世界惯性赛车）
import numpy as np
from collections import defaultdict
from src.utils import info, success
from src.visualize import draw_trajectory, draw_q_value_heatmap
import time

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=1.0, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, step_func=None):
        """
        env: InertialRacingEnv 实例（用于获取网格信息及 step_raw）
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.Q = defaultdict(lambda: np.zeros(env.num_actions))
        self.step_func = step_func if step_func is not None else env.step_raw

    def choose_action(self, state):
        """ε-greedy 动作选择"""
        q_vals = self.Q[state]   # 长度为 num_actions 的数组
        if np.random.random() < self.epsilon:
            # 找出所有 Q 值 > -np.inf 的动作索引（即非死亡动作）
            alive_actions = np.where(q_vals > -np.inf)[0]
            if len(alive_actions) == 0:
                # 所有动作都死亡，那就随便选（其实不可避免死亡）
                return np.random.randint(self.env.num_actions)
            return np.random.choice(alive_actions)
        else:
            # 贪婪选择时也要避免选到 -inf（虽然理论上最优策略不会选）
            # 但若所有动作都是 -inf，则选第一个
            best = np.nanargmax(q_vals)
            if q_vals[best] == -np.inf:
                # 退化情况，随机选一个非死亡动作（如果没有则任意）
                alive = np.where(q_vals > -np.inf)[0]
                if len(alive) > 0:
                    return np.random.choice(alive)
            return best
        
    def update_q(self, state, action, reward, next_state, done):
        """Q-learning 更新公式"""
        best_next = 0.0 if done else np.nanmax(self.Q[next_state])
        target = reward + self.gamma * best_next
        # 如果旧值和目标值都是负无穷，说明这是绝对的死路，不需要更新，直接跳过
        if target == -np.inf and self.Q[state][action] == -np.inf:
            return
        td_error = target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def simulate_greedy_path(self, start_state):
        """
        用当前 Q 表（贪婪策略）从 start_state 模拟一条路径，返回字典（与 trajectory.py 格式兼容）。
        使用 env.step_raw 进行交互。
        """
        path = [start_state[0]]          # 只记录 cur 坐标
        cur, prev = start_state
        total_reward = 0.0
        steps = 0
        success_flag = False
        max_steps = 1000
        
        # 记录已访问的状态，用于检测循环
        visited_states = set()
        visited_states.add((cur, prev))

        while steps < max_steps:
            action = np.nanargmax(self.Q[(cur, prev)])
            next_state, reward, done = self.step_func(cur, prev, action)
            
            # 检测循环：如果下一状态之前已经访问过，说明进入循环
            if not done and next_state is not None:
                if next_state in visited_states:
                    # 检测到循环，标记为失败
                    total_reward = -np.inf
                    success_flag = False
                    break
            
            total_reward += reward

            if done:
                if reward > -np.inf:     # 终点
                    success_flag = True
                    # 需要计算终点坐标（无法从 next_state 获得，因为它是 None）
                    a_vec = self.env.actions[action]
                    vx, vy = cur[0] - prev[0], cur[1] - prev[1]
                    goal = (cur[0] + vx + a_vec[0], cur[1] + vy + a_vec[1])
                    path.append(goal)
                # 死亡不加任何坐标
                break
            else:
                next_cur, next_prev = next_state
                path.append(next_cur)
                cur, prev = next_cur, next_prev
                visited_states.add((cur, prev))  # 记录新状态
                steps += 1

        return {
            'path': path,
            'total_reward': total_reward,
            'steps': steps,
            'success': success_flag,
            'start_coord': start_state[0]
        }


def train_q_learning(env, num_episodes=50000, alpha=0.1, epsilon_decay=0.995, epsilon_min=0.01, slices_num=12, start_states=None, show_process_images=False, return_stats=False):
    """
    训练 Q-learning 智能体并记录过程数据。

    参数:
        env:                   InertialRacingEnv 实例
        num_episodes:          训练幕数
        alpha:                 学习率
        epsilon_decay:         ε衰减因子
        epsilon_min:           ε最小值
        slices_num:            可选，指定切片记录的个数
        start_states:          可选，指定起点列表（格式 [(cur,prev),...]）；
                            若为 None，则使用 env.start_states
        show_process_images:   可选，是否显示过程图片
        return_stats:          可选，是否返回统计信息

    返回:
        agent:                 训练后的 QLearningAgent 实例
        episode_rewards:       每幕的总回报列表
        recorded_paths:        记录时间点的贪婪路径列表（每个元素为 dict，同 simulate_greedy_path 格式）
        record_episodes:       记录时对应的幕编号
        global_best_path_info: 全局最优路径信息
        q_snapshots:           记录时间点的 Q 表列表
    """
    info("开始无模型 Q-learning 训练...")

    # 记录开始时间
    start_time = time.time()

    if start_states is None:
        start_states = env.start_states

    # 提取 BFS 预计算的结果，构造快速 step 函数
    state_to_idx = env.state_to_idx
    transitions = env.transitions
    actions = env.actions
    idx_to_state = env.idx_to_state   # 用于根据索引获取 cur

    def fast_step(cur, prev, action_idx):
        """完全等价于 env.step_raw，但直接查表"""
        state_idx = state_to_idx[(cur, prev)]
        typ, data = transitions[state_idx][action_idx]
        if typ == 'death':
            return None, -np.inf, True
        elif typ == 'goal':
            reward = -10.0 + data
            return None, reward, True
        else:  # normal
            next_idx = data
            next_cur, _ = idx_to_state[next_idx]
            return (next_cur, cur), -10.0, False

    # 创建 agent，注入 fast_step
    agent = QLearningAgent(env, alpha=alpha, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min, step_func=fast_step)
    episode_rewards = []
    recorded_paths = []
    record_episodes = []
    q_snapshots = []

    num_episodes += 1

    if num_episodes <= slices_num:
        # 历史长度不足，返回全部
        indices = list(range(num_episodes))
    else:
        # 均匀采样：包含首尾
        # np.linspace 生成 num_slices 个均匀分布的点
        indices = np.linspace(0, num_episodes - 1, slices_num, dtype=int)
        # 去重并保持顺序（防止因浮点误差导致重复）
        indices = sorted(set(indices.tolist()))

    global_best_path_info = None
    global_best_reward = -np.inf

    for ep in range(0, num_episodes):
        # 随机选一个起点开始本幕（或固定第一个，这里选随机增加多样性）
        start = start_states[np.random.randint(len(start_states))]
        cur, prev = start
        total_reward = 0.0
        done = False
        max_steps = 1000
        step = 0

        while not done and step < max_steps:
            state = (cur, prev)
            action = agent.choose_action(state)
            next_state, reward, done = fast_step(cur, prev, action)
            agent.update_q(state, action, reward, next_state, done)
            total_reward += reward
            if not done:
                cur, prev = next_state
            step += 1

        episode_rewards.append(total_reward)
        agent.decay_epsilon()

        # 记录贪婪路径：遍历所有起点，选择最优路径
        best_path_info = None
        best_reward = -np.inf
        
        # 遍历所有起点，找到最优路径
        if ep in indices:
            for rep_start in start_states:
                path_info = agent.simulate_greedy_path(rep_start)
                if best_path_info is None:
                    best_path_info = path_info
                    best_reward = path_info['total_reward']
                elif path_info['total_reward'] == -np.inf:
                    if best_reward == -np.inf and path_info['steps'] > best_path_info['steps']:
                        best_path_info = path_info
                        best_reward = path_info['total_reward']
                elif path_info['total_reward'] > best_reward:
                    best_path_info = path_info
                    best_reward = path_info['total_reward']
            
            if global_best_path_info is None:
                global_best_path_info = best_path_info
                global_best_reward = best_reward
            elif best_reward == -np.inf:
                if global_best_reward == -np.inf and best_reward > global_best_reward:
                    global_best_path_info = best_path_info
                    global_best_reward = best_reward
            elif best_reward > global_best_reward:
                global_best_path_info = best_path_info
                global_best_reward = best_reward

            q_copy = {k: v.copy() for k, v in agent.Q.items()}
            q_snapshots.append(q_copy)
            
            recorded_paths.append(best_path_info)
            record_episodes.append(ep)

            info(f"Episode {ep:{len(str(num_episodes - 1))}d}/{num_episodes - 1} | ε={agent.epsilon:.3f} | "
                    f"当前贪婪代价={-best_path_info['total_reward']:.1f}")
            
            if show_process_images:
                draw_trajectory(env, best_path_info, title=f"Episode {ep:{len(str(num_episodes - 1))}d}/{num_episodes - 1} | ε={agent.epsilon:.3f}")
                draw_q_value_heatmap(env, agent.Q, title=f"Episode {ep:{len(str(num_episodes - 1))}d}/{num_episodes - 1} | ε={agent.epsilon:.3f}")

    success("Q-learning 训练完成！")

    # 计算训练耗时
    elapsed_ms = (time.time() - start_time) * 1000

    # 收集最优路径信息
    if global_best_path_info is not None:
        best_path_cost = -global_best_path_info['total_reward'] if global_best_path_info['total_reward'] != -np.inf else float('inf')
        best_path_success = global_best_path_info['success']
        best_path_steps = global_best_path_info['steps']
    else:
        best_path_cost = float('inf')
        best_path_success = False
        best_path_steps = None

    stats = {
        'total_episodes': num_episodes,
        'final_epsilon': agent.epsilon,
        'best_path_cost': best_path_cost,
        'best_path_success': best_path_success,
        'best_path_steps': best_path_steps,
        'training_time_ms': elapsed_ms,
    }

    info(f"Q-learning 统计: 最终 ε={agent.epsilon:.4f}, 最优代价={best_path_cost}, 成功={best_path_success}, 耗时={elapsed_ms:.2f}ms")

    if return_stats:
        return agent, episode_rewards, recorded_paths, record_episodes, global_best_path_info, q_snapshots, stats
    else:
        return agent, episode_rewards, recorded_paths, record_episodes, global_best_path_info, q_snapshots
