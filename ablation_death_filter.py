# ablation_death_filter.py
import numpy as np
import time
import csv
import os
from collections import defaultdict
from src.track_loader import load_track
from src.environment import InertialRacingEnv
from src.utils import setup_logging, info, success, warning

setup_logging()

# ---------- 自定义 QLearningAgent 支持 filter_dead ----------
class QLearningAgentWithFilter:
    def __init__(self, env, alpha=1.0, gamma=1.0, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995, step_func=None, filter_dead=True):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.filter_dead = filter_dead
        self.Q = defaultdict(lambda: np.zeros(env.num_actions))
        self.step_func = step_func if step_func is not None else env.step_raw

    def choose_action(self, state):
        q_vals = self.Q[state]
        if np.random.random() < self.epsilon:
            if self.filter_dead:
                # 只从非死亡动作中选
                alive = np.where(q_vals > -np.inf)[0]
                if len(alive) == 0:
                    alive = list(range(len(q_vals)))
                return np.random.choice(alive)
            else:
                # 允许选任何动作，包括死亡动作
                return np.random.randint(len(q_vals))
        else:
            # 贪婪：避免选到 -inf（如果可能）
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
        """返回 (total_reward, steps, success)"""
        cur, prev = start_state
        total_reward = 0.0
        steps = 0
        max_steps = 1000
        visited = set()
        visited.add((cur, prev))
        while steps < max_steps:
            state = (cur, prev)
            q_vals = self.Q[state]
            # 避免选 -inf
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


def train_q_learning_early_stop(env, num_episodes=100000, alpha=1.0, epsilon_decay=0.995,
                                epsilon_min=0.1, filter_dead=True, target_cost=100.0,
                                start_states=None):
    """
    训练 Q-learning，当贪婪路径代价达到 target_cost 时提前停止。
    返回 (converged, episodes_used, time_ms, final_cost)
    """
    if start_states is None:
        start_states = env.start_states

    # 快速 step 函数（复用环境预计算）
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

    agent = QLearningAgentWithFilter(env, alpha=alpha, epsilon_decay=epsilon_decay,
                                     epsilon_min=epsilon_min, step_func=fast_step,
                                     filter_dead=filter_dead)

    train_time = 0.0   # 累计训练时间（毫秒）
    best_cost = float('inf')
    converged = False
    episodes_used = 0

    for ep in range(num_episodes):
        # 随机起点
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

        # 每幕检查一次最优路径（不计时）
        if ep % 1 == 0 or ep == num_episodes - 1:
            # 评估所有起点，找最优代价
            current_best_cost = float('inf')
            for s in start_states:
                total_reward, _, success = agent.simulate_greedy_path(s)
                if success and -total_reward < current_best_cost:
                    current_best_cost = -total_reward
            if current_best_cost < best_cost:
                best_cost = current_best_cost
                info(f"Episode {ep}: new best cost = {best_cost:.1f}")
            if best_cost <= target_cost:
                converged = True
                episodes_used = ep + 1
                break

    elapsed_ms = train_time   # 只包含训练时间
    if not converged:
        episodes_used = num_episodes
        warning(f"未收敛，最终代价 {best_cost}")
    return converged, episodes_used, elapsed_ms, best_cost


# ---------- 主实验 ----------
def run_ablation():
    track_name = "U_shape_with_narrow_shortcut"
    target_cost = 100.0
    repeats = 50
    max_episodes = 50000  # 足够大，确保能收敛

    # 加载环境
    grid = load_track(f"tracks/{track_name}.txt")
    env = InertialRacingEnv(grid)

    results = []
    for filter_dead in [True, False]:
        mode = "filter" if filter_dead else "no_filter"
        info(f"===== 运行模式: {mode} =====")
        for rep in range(repeats):
            info(f"重复 {rep+1}/{repeats}")
            converged, episodes, time_ms, final_cost = train_q_learning_early_stop(
                env, num_episodes=max_episodes, alpha=1.0, epsilon_min=0.1,
                epsilon_decay=0.995, filter_dead=filter_dead,
                target_cost=target_cost, start_states=env.start_states
            )
            results.append({
                "mode": mode,
                "repeat": rep+1,
                "converged": converged,
                "episodes_used": episodes,
                "time_ms": time_ms,
                "final_cost": final_cost
            })
            info(f"  收敛: {converged}, 幕数: {episodes}, 时间: {time_ms:.0f}ms")

    # 保存结果到 CSV
    csv_file = "logs/ablation_death_filter.csv"
    os.makedirs("logs", exist_ok=True)
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["mode", "repeat", "converged", "episodes_used", "time_ms", "final_cost"])
        writer.writeheader()
        writer.writerows(results)

    # 打印统计摘要
    import pandas as pd
    df = pd.DataFrame(results)
    summary = df.groupby("mode").agg({
        "episodes_used": ["mean", "std"],
        "time_ms": ["mean", "std"]
    }).round(0)
    print("\n===== 消融实验结果 =====")
    print(summary)
    success(f"结果已保存至 {csv_file}")


if __name__ == "__main__":
    run_ablation()