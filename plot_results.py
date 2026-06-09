# plot_results.py

"""
plot_results.py - 生成论文所需所有图表
数据源：
- tracks/track_data.csv
- logs/run_all.csv
- logs/convergence_ql.csv
- logs/ablation_death_filter.csv
输出目录：summary_figs/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = "summary_figs"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------- 1. 合并图：VI + QL 收敛步数 vs 可行驶格子数 -----------------------------
# ----------------------------- 1. 价值迭代单独图 -----------------------------
print("绘制图1a: 价值迭代收敛轮数 vs 可行驶格子数 (单独)")
track_info = pd.read_csv("tracks/track_data.csv", encoding='utf-8')
run_all = pd.read_csv("logs/run_all.csv")
vi_df = run_all[run_all["algorithm"] == "Value Iteration"][["track", "iterations"]]
df_vi = pd.merge(track_info, vi_df, left_on="赛道名称", right_on="track", how="left")
df_vi = df_vi.rename(columns={"可行驶格子数": "grids", "iterations": "vi_iters", "赛道名称": "track_name"})

plt.figure(figsize=(6, 4))
plt.scatter(df_vi["grids"], df_vi["vi_iters"], color='navy', s=60, edgecolors='white', alpha=0.8)
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Number of Drivable Grids", fontsize=12)
plt.ylabel("Value Iteration Iterations", fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.title("Value Iteration Convergence Iterations vs State Space Size", fontsize=12)
for i, row in df_vi.iterrows():
    plt.annotate(row["track_name"], (row["grids"], row["vi_iters"]),
                 textcoords="offset points", xytext=(5,5), fontsize=8, alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_vi_vs_grids.png"), dpi=300)
plt.close()

# ----------------------------- 2. Q-learning 单独图 -----------------------------
print("绘制图1b: Q-learning 收敛幕数 vs 可行驶格子数 (单独)")
ql_early = pd.read_csv("logs/convergence_ql.csv")
ql_mean_ep = ql_early.groupby("track")["episodes_used"].mean().reset_index()
ql_mean_ep = ql_mean_ep.rename(columns={"episodes_used": "ql_episodes"})
df_ql = pd.merge(track_info, ql_mean_ep, left_on="赛道名称", right_on="track", how="left")
df_ql = df_ql.rename(columns={"可行驶格子数": "grids", "赛道名称": "track_name"})

plt.figure(figsize=(6, 4))
plt.scatter(df_ql["grids"], df_ql["ql_episodes"], color='darkorange', s=60, edgecolors='white', alpha=0.8, marker='s')
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Number of Drivable Grids", fontsize=12)
plt.ylabel("Q-learning Episodes to Convergence (mean)", fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.title("Q-learning Convergence Episodes vs State Space Size", fontsize=12)
for i, row in df_ql.iterrows():
    plt.annotate(row["track_name"], (row["grids"], row["ql_episodes"]),
                 textcoords="offset points", xytext=(5,5), fontsize=8, alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_ql_vs_grids.png"), dpi=300)
plt.close()

# ----------------------------- 3. 合并图 -----------------------------
print("绘制图1c: 价值迭代与 Q-learning 合并散点图")
plt.figure(figsize=(7, 10))
plt.scatter(df_vi["grids"], df_vi["vi_iters"], color='navy', s=60, edgecolors='white', alpha=0.8, label='Value Iteration')
plt.scatter(df_ql["grids"], df_ql["ql_episodes"], color='darkorange', s=60, edgecolors='white', alpha=0.8, marker='s', label='Q-learning')
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Number of Drivable Grids", fontsize=12)
plt.ylabel("Convergence Steps (Iterations / Episodes)", fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.title("Convergence Complexity: Model-based vs Model-free", fontsize=14)
# 标注（为避免拥挤，可只标部分）
for i, row in df_vi.iterrows():
    plt.annotate(row["track_name"], (row["grids"], row["vi_iters"]),
                 textcoords="offset points", xytext=(5,5), fontsize=8, alpha=0.7, color='navy')
for i, row in df_ql.iterrows():
    plt.annotate(row["track_name"], (row["grids"], row["ql_episodes"]),
                 textcoords="offset points", xytext=(5,-12), fontsize=8, alpha=0.7, color='darkorange')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_convergence_vs_grids.png"), dpi=300)
plt.close()

# ----------------------------- 2. Q-learning 早停实验收敛幕数箱线图（对数纵轴） -----------------------------
print("绘制图2: Q-learning 收敛幕数箱线图 (对数纵轴)")
order = ["toy", "short_straight", "straight", "U_shape", "U_shape_with_narrow_shortcut",
         "S_shape", "variant_U_shape", "random"]
plt.figure(figsize=(6, 9))
sns.boxplot(data=ql_early, x="track", y="episodes_used", order=order, palette="viridis")
sns.stripplot(data=ql_early, x="track", y="episodes_used", order=order,
              color="black", alpha=0.4, size=3, jitter=True)
plt.yscale('log')
plt.xticks(rotation=45, ha='right')
plt.xlabel("Track", fontsize=12)
plt.ylabel("Episodes to Convergence (log scale)", fontsize=12)
plt.title("Q-learning Convergence Episodes (early stop, 10 runs)", fontsize=14)
plt.grid(axis='y', which='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_ql_convergence_episodes_box.png"), dpi=300)
plt.close()

# ----------------------------- 3. Q-learning 收敛时间箱线图（对数纵轴） -----------------------------
print("绘制图3: Q-learning 收敛时间箱线图 (对数纵轴)")
plt.figure(figsize=(6, 9))
sns.boxplot(data=ql_early, x="track", y="time_ms", order=order, palette="viridis")
sns.stripplot(data=ql_early, x="track", y="time_ms", order=order,
              color="black", alpha=0.4, size=3, jitter=True)
plt.yscale('log')
plt.xticks(rotation=45, ha='right')
plt.xlabel("Track", fontsize=12)
plt.ylabel("Time to Convergence (ms, log scale)", fontsize=12)
plt.title("Q-learning Convergence Time (early stop, 10 runs)", fontsize=14)
plt.grid(axis='y', which='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_ql_convergence_time_box.png"), dpi=300)
plt.close()

# ----------------------------- 4. 价值迭代 vs Q-learning 收敛步数对比柱状图 -----------------------------
print("绘制图4: 价值迭代 vs Q-learning 收敛步数对比柱状图")
vi_conv = df_vi[["track_name", "vi_iters"]].rename(columns={"track_name": "track", "vi_iters": "vi_iter"})
ql_summary = pd.read_csv("logs/convergence_ql_summary.csv")  # 包含 conv_mean, conv_std
compare = pd.merge(vi_conv, ql_summary, on="track", how="inner")
compare = compare.sort_values("track", key=lambda x: x.map({t: i for i, t in enumerate(order)}))

x = np.arange(len(compare))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width/2, compare["vi_iter"], width, label='Value Iteration (iterations)', color='steelblue')
ax.bar(x + width/2, compare["conv_mean"], width, label='Q-learning (episodes)', color='salmon',
       yerr=compare["conv_std"], capsize=3)
ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(compare["track"], rotation=45, ha='right')
ax.set_ylabel("Number of Iterations / Episodes (log scale)", fontsize=12)
ax.set_title("Convergence Complexity: Model-based vs Model-free", fontsize=14)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_vi_vs_ql_convergence.png"), dpi=300)
plt.close()

# ----------------------------- 5. 消融实验 -----------------------------
print("绘制图5: 消融实验收敛幕数箱线图")
ablation = pd.read_csv("logs/ablation_death_filter.csv")
plt.figure(figsize=(4, 6))
sns.boxplot(data=ablation, x="mode", y="episodes_used", palette={"filter": "#2ecc71", "no_filter": "#e74c3c"})
sns.stripplot(data=ablation, x="mode", y="episodes_used", color="black", alpha=0.5, jitter=True)
plt.xlabel("Exploration Mode", fontsize=12)
plt.ylabel("Episodes to Convergence", fontsize=12)
plt.title("Effect on Episodes\nof Death Action Filtering\n(on narrow shortcut track)", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_ablation_episodes.png"), dpi=300)
plt.close()

print("绘制图6: 消融实验收敛时间箱线图")
plt.figure(figsize=(4, 6))
sns.boxplot(data=ablation, x="mode", y="time_ms", palette={"filter": "#2ecc71", "no_filter": "#e74c3c"})
sns.stripplot(data=ablation, x="mode", y="time_ms", color="black", alpha=0.5, jitter=True)
plt.xlabel("Exploration Mode", fontsize=12)
plt.ylabel("Time to Convergence (ms)", fontsize=12)
plt.title("Effect on Computation Time\nof Death Action Filtering\n(on narrow shortcut track)", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_ablation_time.png"), dpi=300)
plt.close()

# ----------------------------- 6. 状态空间压缩率 -----------------------------
print("绘制图7: 状态空间压缩率")
track_info_sorted = track_info.sort_values("可行驶格子数")
x = np.arange(len(track_info_sorted))
width = 0.35
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.bar(x - width/2, track_info_sorted["理论状态数 (格子数²)"], width, label='Theoretical States', color='lightblue')
ax1.bar(x + width/2, track_info_sorted["BFS合法状态数"], width, label='Reachable States', color='orange')
ax1.set_yscale('log')
ax1.set_xlabel("Track", fontsize=12)
ax1.set_ylabel("Number of States (log scale)", fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(track_info_sorted["赛道名称"], rotation=45, ha='right')
ax1.legend(loc='upper left')
compression = track_info_sorted["压缩率"].str.rstrip('%').astype(float)
ax2 = ax1.twinx()
ax2.plot(x, compression, 'ro-', label='Compression Rate (%)')
ax2.set_ylabel("Compression Rate (%)", fontsize=12, color='red')
ax2.tick_params(axis='y', labelcolor='red')
# legend 放到外面
ax2.legend(loc='upper right', bbox_to_anchor=(1, 1.1))
plt.title("State Space Compression by BFS", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_state_compression.png"), dpi=300)
plt.close()

print(f"\n所有图表生成完成，保存在 {OUT_DIR}/")