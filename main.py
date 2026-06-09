# main.py
from src.track_loader import load_track
from src.environment import InertialRacingEnv
from src.value_iteration import value_iteration
from src.q_learning import train_q_learning
from src.trajectory import find_best_start_and_path, analyze_policy_slices_paths
from src.history_utils import sample_history_slices
from src.visualize import *
from src.utils import Timer, setup_logging
import csv
import os
import time

setup_logging()

# 写入csv
def write_to_csv(filename, start_time, track_name, algorithm, repeat_num, converged, iterations, best_cost, time_ms):
    # ---------- 写入 CSV ----------
    os.makedirs("logs", exist_ok=True)
    csv_file = f"logs/{filename}.csv"
    file_exists = os.path.isfile(csv_file)

    row = {
        "start_time": start_time,
        "track": track_name,
        "algorithm": algorithm,
        "repeat_num": repeat_num,
        "converged": converged,
        "iterations": iterations,
        "best_cost": best_cost,
        "time_ms": time_ms,
    }

    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main(RUNMARK = "defalt", track_name = "toy", repeat_num = 0, only_best_path = False, run_vi = True, run_ql = False, ql_episodes = 100000, do_plotting = True, save_to_file = True):
    '''
    :param track_name:     赛道名称
    :param only_best_path: 是否仅计算最优路径
    :param run_vi:         是否运行价值迭代
    :param run_ql:         是否运行 Q-learning
    :param ql_episodes:    如果运行 Q-learning，运行的 episode 数
    :param do_plotting:    是否进行绘图
    :param save_to_file:   如果进行绘图，是否保存图片（开启时保存图片，关闭时会显示图片）
    '''
    
    ## 读取赛道
    grid = load_track("tracks/"+track_name+".txt")
    
    ## 创建环境
    env = InertialRacingEnv(grid)

    if do_plotting:
        draw_track(env, title=track_name+"赛道布局", add_padding=True, save_to_file=save_to_file)
    
    # ========== 有模型价值迭代 ==========
    ## 价值迭代
    if run_vi:
        # 记录开始时间（完整格式 YYYY-MM-DD HH:MM:SS）
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        with Timer(f"[{track_name}] 价值迭代 (直至收敛)"):
            if only_best_path:
                V, policy, vi_stats = value_iteration(env, record_history=False, return_stats=True)
            else:
                V, policy, V_history, policy_history, vi_stats = value_iteration(env, record_history=True, return_stats=True)

        if do_plotting:
            # 绘制状态价值热力图
            draw_value_heatmap(env, V, title=track_name+"单元格价值热力图", save_to_file=save_to_file)

        ## 寻找最优路径
        if only_best_path:
            best_path = find_best_start_and_path(env, policy, return_all=False)
        else:
            best_path, all_paths = find_best_start_and_path(env, policy, return_all=True)

        if do_plotting:
            draw_trajectory(env, best_path, title=track_name+"最优轨迹", add_padding=True, save_to_file=save_to_file)
            if not only_best_path:
                draw_trajectory(env, all_paths, title=track_name+"各起点最优轨迹", add_padding=True, save_to_file=save_to_file)
                ## 迭代历史切片
                V_slices, policy_slices, indices = sample_history_slices(V_history, policy_history)
                ## 绘制价值迭代过程（价值函数演变）
                draw_value_heatmap_grid(env, V_slices, indices, title=track_name+"单元格价值热力演化图", save_to_file=save_to_file)
                ## 绘制价值迭代过程（策略轨迹演进）
                best_path_slices = analyze_policy_slices_paths(env, policy_slices)
                draw_trajectory_grid(env, best_path_slices, indices, title=track_name+"最优轨迹演化图", save_to_file=save_to_file)
        
        write_to_csv(RUNMARK, start_time, track_name, "Value Iteration", repeat_num, vi_stats['converged'], vi_stats['iterations'], -best_path['total_reward'], vi_stats['time_ms'])

    # ========== 无模型 Q-learning ==========
    if run_ql and not only_best_path: # 只在非快速模式下运行
        # 记录开始时间（完整格式 YYYY-MM-DD HH:MM:SS）
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        # 超参数
        alpha = 1.0
        epsilon_min = 0.1
        epsilon_decay = epsilon_min ** (1 / (0.9 * ql_episodes))

        with Timer(f"[{track_name}] Q-learning 训练 ({ql_episodes} episodes)"):
            agent, ep_rewards, q_recorded_paths, q_episodes, global_best_path_info, q_snapshots, ql_stats = train_q_learning(
                env, num_episodes=ql_episodes, alpha=alpha, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min, 
                show_process_images=False,
                slices_num=12, start_states=None, 
                return_stats=True)

        if do_plotting:
            # 绘制 Q-learning 最终热力图
            draw_q_value_heatmap(env, agent.Q, title=f"{track_name} 最终 Q-learning 热力图",
                                save_to_file=save_to_file)
        
            # 绘制 Q-learning 最终轨迹
            draw_trajectory(env, global_best_path_info, title=f"{track_name} 最佳 Q-learning 轨迹", add_padding=True,
                                save_to_file=save_to_file)
        
            # 绘制学习曲线
            draw_q_learning_curve(ep_rewards,
                                title=track_name+" Q-learning 学习曲线",
                                save_to_file=save_to_file, window=None)

            # 绘制探索过程中的轨迹演化（复用已有的网格图函数）
            draw_trajectory_grid(
                env, q_recorded_paths, q_episodes,
                title=track_name+" Q-learning 轨迹演化",
                add_padding=True, arrow_linewidth=2.0, save_to_file=save_to_file
            )

            # 绘制 Q-learning 价值热力图演化
            draw_q_value_heatmap_grid(
                env, q_snapshots, q_episodes,
                title=f"{track_name} Q-learning 价值演化",
                share_colorbar=True, save_to_file=save_to_file
            )

        write_to_csv(RUNMARK, start_time, track_name, "Q-Learning", repeat_num, "N/A", ql_stats['total_episodes'], ql_stats['best_path_cost'], ql_stats['training_time_ms'])


if __name__ == "__main__":
    # 所有轨道
    all_tracks = [
        "toy", "straight", "short_straight", "U_shape", "U_shape_with_narrow_shortcut",
        "S_shape", "variant_U_shape", "random", "custom"
    ]

    # 价值迭代运行次数
    vi_run_times = {
        "toy":                          1, # 1
        "straight":                     1, # 1
        "short_straight":               1, # 1
        "U_shape":                      1, # 1
        "U_shape_with_narrow_shortcut": 1, # 1
        "S_shape":                      1, # 1
        "variant_U_shape":              1, # 1
        "random":                       1, # 1
        "custom":                       1, # 1
    }

    # Q-learning 运行次数
    ql_run_times = {
        "toy":                          10,  # 10
        "straight":                     5,   # 5
        "short_straight":               10,  # 10
        "U_shape":                      5,   # 5
        "U_shape_with_narrow_shortcut": 5,   # 5
        "S_shape":                      5,   # 5
        "variant_U_shape":              3,   # 3
        "random":                       3,   # 3
        "custom":                       1,   # 1
    }

    # 预计算的单词运行 Q-learning 的 episode 数
    precomputed_num_episodes = {
        "toy":                          300,
        "straight":                     8000,
        "short_straight":               300,
        "U_shape":                      30000,
        "U_shape_with_narrow_shortcut": 35000,
        "S_shape":                      50000,
        "variant_U_shape":              150000,
        "random":                       200000,
        "custom":                       2000000,
    }

    RUNMARK = "run_all"

    # 按照逐轨道的顺序运行
    for track_name in all_tracks:
        for i in range(vi_run_times[track_name]):
            main(RUNMARK = RUNMARK, track_name = track_name, repeat_num = i+1, only_best_path = False, run_vi = True, run_ql = False, ql_episodes = 0, do_plotting = False, save_to_file = False)
        
        for i in range(ql_run_times[track_name]):
            main(RUNMARK = RUNMARK, track_name = track_name, repeat_num = i+1, only_best_path = False, run_vi = False, run_ql = True, ql_episodes = precomputed_num_episodes[track_name], do_plotting = False, save_to_file = False)
    
    # # 单独运行以绘图
    # track_name = "short_straight"
    # main(track_name = track_name, run_vi = True, run_ql = True, ql_episodes = precomputed_num_episodes[track_name], do_plotting = True, save_to_file = True)
