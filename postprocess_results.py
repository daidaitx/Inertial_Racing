# postprocess_results.py
"""
后处理脚本：统计价值迭代和 Q-learning 的实验结果
输入：
  - logs/run_all.csv : 原始实验结果
  - logs/track_data.csv : 赛道静态数据（可选）
输出：
  - logs/results_summary.csv : 汇总统计表
  - 控制台打印美观表格
"""

import pandas as pd
import numpy as np
import os

# 文件路径
run_all_path = "logs/run_all.csv"
track_data_path = "tracks/track_data.csv"  # 如果存在
output_path = "results_summary.csv"

# 读取数据
df = pd.read_csv(run_all_path)

# 分离价值迭代和 Q-learning
df_vi = df[df["algorithm"] == "Value Iteration"].copy()
df_ql = df[df["algorithm"] == "Q-Learning"].copy()

# 价值迭代：每个赛道只有一行
vi_summary = df_vi[["track", "best_cost", "time_ms"]].rename(
    columns={"best_cost": "vi_best_cost", "time_ms": "vi_time_ms"}
)

# Q-learning 统计：按赛道分组
ql_stats = df_ql.groupby("track").agg(
    ql_repeats=("repeat_num", "count"),
    ql_best_cost_mean=("best_cost", "mean"),
    ql_best_cost_std=("best_cost", "std"),
    ql_time_ms_mean=("time_ms", "mean"),
    ql_time_ms_std=("time_ms", "std")
).reset_index()

# 合并价值迭代和 Q-learning 统计
summary = pd.merge(vi_summary, ql_stats, on="track", how="left")

# 可选：合并赛道静态数据
if os.path.exists(track_data_path):
    track_info = pd.read_csv(track_data_path)
    # 统一列名：假设 track_data.csv 有 "赛道名称" 列，改成 "track"
    if "赛道名称" in track_info.columns:
        track_info = track_info.rename(columns={"赛道名称": "track"})
    summary = pd.merge(summary, track_info, on="track", how="left")

# 重新排列列顺序
cols = [
    "track",
    "vi_best_cost", "vi_time_ms",
    "ql_repeats",
    "ql_best_cost_mean", "ql_best_cost_std",
    "ql_time_ms_mean", "ql_time_ms_std"
]
# 如果 track_info 有更多列，可追加在后面
if "尺寸 (H)" in summary.columns:
    cols.extend(["尺寸 (H)", "尺寸 (W)", "可行驶格子数", "理论状态数 (格子数²)", "BFS合法状态数", "压缩率"])

summary = summary[cols]

# 格式化数值：保留 2 位小数，时间保留 0 位小数
summary["vi_time_ms"] = summary["vi_time_ms"].round(0).astype(int)
summary["ql_time_ms_mean"] = summary["ql_time_ms_mean"].round(0).astype(int)
summary["ql_time_ms_std"] = summary["ql_time_ms_std"].apply(
    lambda x: "-" if pd.isna(x) else int(round(x))
)
summary["ql_best_cost_mean"] = summary["ql_best_cost_mean"].round(3)
summary["ql_best_cost_std"] = summary["ql_best_cost_std"].apply(
    lambda x: "-" if pd.isna(x) else round(x, 3)
)

# 保存到 CSV
summary.to_csv(output_path, index=False, encoding="utf-8")
print(f"汇总结果已保存至 {output_path}")

# 打印到控制台（美观表格）
print("\n================== 实验结果汇总 ==================")
print(summary.to_string(index=False))
print("==================================================\n")

# 额外打印一些关键对比（可选）
print("关键对比（价值迭代 vs Q-learning）:")
for _, row in summary.iterrows():
    track = row["track"]
    vi_time = row["vi_time_ms"]
    ql_time_mean = row["ql_time_ms_mean"]
    vi_cost = row["vi_best_cost"]
    ql_cost_mean = row["ql_best_cost_mean"]
    print(f"{track:25} | VI: {vi_time:>6} ms, cost={vi_cost:>7.2f} | QL: {ql_time_mean:>7} ms, cost={ql_cost_mean:>7.2f} (repeats={row['ql_repeats']})")