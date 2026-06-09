"""
对前8个赛道运行 Q-learning 早停实验（达到最优代价即停止）
每 100 幕检测一次，每个赛道重复 10 次。
记录收敛幕数、时间，并输出统计结果。
"""

import numpy as np
import time
import csv
import os
import pandas as pd
from collections import defaultdict
from src.track_loader import load_track
from src.environment import InertialRacingEnv
from src.utils import setup_logging, info, success, warning

setup_logging()

# ========== 配置 ==========
# 赛道列表（排除 custom）
TRACKS = [
    "toy",
    "straight",
    "short_straight",
    "U_shape",
    "U_shape_with_narrow_shortcut",
    "S_shape",
    "variant_U_shape",
    "random",
]

# 每个赛道的最优代价（来自价值迭代结果，见 results_summary.csv）
BEST_COST = {
    "toy": 40.0,
    "straight": 110.0,
    "short_straight": 30.0,
    "U_shape": 105.0,
    "U_shape_with_narrow_shortcut": 100.0,
    "S_shape": 206.25,
    "variant_U_shape": 104.0,
    "random": 137.5,
}

# 每个赛道的最大训练幕数（足够大，确保能收敛）
MAX_EPISODES = {
    "toy": 20000,
    "straight": 20000,
    "short_straight": 5000,
    "U_shape": 50000,
    "U_shape_with_narrow_shortcut": 50000,
    "S_shape": 100000,
    "variant_U_shape": 200000,
    "random": 300000,
}

REPEATS = 10
DETECT_INTERVAL = 1
ALPHA = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.995  # 固定衰减率，与之前实验一致
GAMMA = 1.0

# ========== 自定义 Agent（支持早停，标准过滤模式） ==========
class QLearningAgentEarlyStop:
    def __init__(self, env, alpha=1.0, gamma=1.0, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995, step_func=None):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.Q = defaultdict(lambda: np.zeros(env.num_actions))
        self.step_func = step_func if step_func is not None else env.step_raw

    def choose_action(self, state):
        """ε-greedy，只从非死亡动作中选择（标准模式）"""
        q_vals = self.Q[state]
        if np.random.random() < self.epsilon:
            alive = np.where(q_vals > -np.inf)[0]
            if len(alive) == 0:
                alive = list(range(len(q_vals)))
            return np.random.choice(alive)
        else:
            best = np.nanargmax(q_vals)
            if q_vals[best] == -np.inf:
                alive = np.where(q_vals > -np.inf)[0]
                if len(alive) > 0:
                    return np.random.choice(alive)
            return best

    def update_q(self, state, action, reward, next_state, done):
        best_next = 0.0 if done else np.nanmax(self.Q[next_state])
        target = reward + self.gamma * best_next
        if target == -np.inf and self.Q[state][action] == -np.inf:
            return
        td_error = target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def simulate_greedy_path(self, start_state):
        """模拟贪婪路径，返回 (total_reward, steps, success)"""
        cur, prev = start_state
        total_reward = 0.0
        steps = 0
        max_steps = 1000
        visited = set()
        visited.add((cur, prev))
        while steps < max_steps:
            state = (cur, prev)
            q_vals = self.Q[state]
            best = np.nanargmax(q_vals)
            if q_vals[best] == -np.inf:
                alive = np.where(q_vals > -np.inf)[0]
                if len(alive) == 0:
                    return -np.inf, steps, False
                action = np.random.choice(alive)
            else:
                action = best
            next_state, reward, done = self.step_func(cur, prev, action)
            total_reward += reward
            steps += 1
            if done:
                if reward > -np.inf:
                    return total_reward, steps, True
                else:
                    return -np.inf, steps, False
            else:
                cur, prev = next_state
                if (cur, prev) in visited:
                    return -np.inf, steps, False
                visited.add((cur, prev))
        return -np.inf, steps, False

    def get_best_cost(self, start_states):
        """返回当前最优代价（所有起点中的最小代价）"""
        best = float('inf')
        for s in start_states:
            total_reward, _, success = self.simulate_greedy_path(s)
            if success:
                cost = -total_reward
                if cost < best:
                    best = cost
        return best


def train_early_stop(env, target_cost, max_episodes, detect_interval=100):
    """
    训练 Q-learning，每 detect_interval 幕检测一次最优代价，
    达到 target_cost 时停止。
    返回 (converged, episodes_used, time_ms, final_cost)
    """
    start_states = env.start_states
    state_to_idx = env.state_to_idx
    transitions = env.transitions
    idx_to_state = env.idx_to_state

    def fast_step(cur, prev, action_idx):
        state_idx = state_to_idx[(cur, prev)]
        typ, data = transitions[state_idx][action_idx]
        if typ == 'death':
            return None, -np.inf, True
        elif typ == 'goal':
            reward = -10.0 + data
            return None, reward, True
        else:
            next_idx = data
            next_cur, _ = idx_to_state[next_idx]
            return (next_cur, cur), -10.0, False

    agent = QLearningAgentEarlyStop(env, alpha=ALPHA, gamma=GAMMA,
                                    epsilon=1.0, epsilon_min=EPSILON_MIN,
                                    epsilon_decay=EPSILON_DECAY,
                                    step_func=fast_step)

    train_time = 0.0
    converged = False
    episodes_used = 0
    final_cost = float('inf')

    for ep in range(1, max_episodes + 1):  # 从1开始，方便计算
        # 一个 episode (计时)
        start = start_states[np.random.randint(len(start_states))]
        cur, prev = start
        done = False
        step = 0
        t0 = time.time()
        while not done and step < 1000:
            state = (cur, prev)
            action = agent.choose_action(state)
            next_state, reward, done = fast_step(cur, prev, action)
            agent.update_q(state, action, reward, next_state, done)
            if not done:
                cur, prev = next_state
            step += 1
        train_time += (time.time() - t0) * 1000
        agent.decay_epsilon()

        # 每 detect_interval 幕检测一次 （不计时）
        if ep % detect_interval == 0:
            current_best = agent.get_best_cost(start_states)
            if current_best < final_cost:
                final_cost = current_best
                info(f"Episode {ep}: new best cost = {final_cost:.2f}")
            if final_cost <= target_cost + 1e-6:  # 允许微小误差
                converged = True
                episodes_used = ep
                break

    elapsed_ms = train_time
    if not converged:
        episodes_used = max_episodes
        warning(f"未收敛，最终代价 {final_cost:.2f}")

    return converged, episodes_used, elapsed_ms, final_cost


# ========== 主实验 ==========
def main():
    results = []

    for track in TRACKS:
        target = BEST_COST[track]
        max_ep = MAX_EPISODES[track]

        info(f"\n===== 赛道: {track} (目标代价 {target}) =====")
        grid = load_track(f"tracks/{track}.txt")
        env = InertialRacingEnv(grid)

        for rep in range(1, REPEATS + 1):
            info(f"重复 {rep}/{REPEATS}")
            conv, episodes, time_ms, final = train_early_stop(env, target, max_ep, DETECT_INTERVAL)
            results.append({
                "track": track,
                "repeat": rep,
                "converged": conv,
                "episodes_used": episodes,
                "time_ms": time_ms,
                "final_cost": final,
            })
            info(f"  收敛={conv}, 幕数={episodes}, 时间={time_ms:.0f}ms")

    # 保存原始结果
    df = pd.DataFrame(results)
    csv_path = "logs/convergence_ql.csv"
    os.makedirs("logs", exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    success(f"原始结果已保存至 {csv_path}")

    # 统计聚合
    summary = df.groupby("track").agg(
        conv_mean=("episodes_used", "mean"),
        conv_std=("episodes_used", "std"),
        time_mean=("time_ms", "mean"),
        time_std=("time_ms", "std"),
        success_rate=("converged", "mean")
    ).round(0)

    # 合并目标代价
    summary["target_cost"] = [BEST_COST[t] for t in summary.index]

    # 输出汇总
    print("\n================== Q-learning 早停实验汇总 ==================")
    print(summary)
    summary.to_csv("logs/convergence_ql_summary.csv", encoding="utf-8")
    success("汇总结果已保存至 logs/convergence_ql_summary.csv")


if __name__ == "__main__":
    main()