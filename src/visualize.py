# src/visualize.py
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']   # 指定默认字体为黑体 (SimHei)
plt.rcParams['axes.unicode_minus'] = False     # 解决保存图像时负号'-'显示为方块的问题
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import os
from src.utils import image_saved, info, warning

# ---------- 工具函数：为数组添加灰色边框 ----------
def pad_with_gray(grid, pad_value=4):
    """在 grid 四周添加一圈灰色边框，灰色用整数 4 表示"""
    h, w = grid.shape
    padded = np.full((h+2, w+2), pad_value, dtype=grid.dtype)
    padded[1:-1, 1:-1] = grid
    return padded

def pad_heatmap_with_nan(heat):
    """热力图边框用 NaN，后续通过 set_bad 设为灰色"""
    h, w = heat.shape
    padded = np.full((h+2, w+2), np.nan, dtype=float)
    padded[1:-1, 1:-1] = heat
    return padded

def get_track_colormap():
    """自定义赛道 colormap：普通/起点/终点/墙 + 灰色边框"""
    colors = ['white', 'lime', 'red', 'black', 'lightgray']
    return ListedColormap(colors)

# ---------- 辅助函数：生成智能偏移点 ----------
def _generate_sorted_offsets(n):
    """
    根据轨迹数量 n 生成 n 个偏移点，按距离中心从近到远排序
    
    参数:
        n: 轨迹数量
    
    返回:
        list of (dx, dy) 元组，已按距离中心从近到远排序
    
    算法:
        1. 计算最小网格边长 m = ceil(sqrt(n))
        2. 扩大网格 m' = m + 2（留出边缘空位）
        3. 在 [-0.45, +0.45] 范围内生成 m'×m' 均匀网格
        4. 按到中心的欧氏距离排序
        5. 取前 n 个点（自然选取最靠近中心的 n 个点）
    
    优势:
        - n 较小时轨迹更集中在中心区域
        - n 较大时自动扩展分布范围
        - 避免少量轨迹过于分散
    """
    if n == 1:
        return [(0.0, 0.0)]
    
    import numpy as np
    
    # 步骤1：计算最小网格边长
    m = int(np.ceil(np.sqrt(n)))  # 最小满足 m² ≥ n 的整数
    
    if m == 1:
        return [(0.0, 0.0)]
    
    # 步骤2：扩大网格（多一圈，让少量轨迹更集中）
    m_prime = m + 2
    
    # 步骤3：生成均匀网格点
    step = 0.9 / (m_prime - 1)  # 总宽度 0.9 (±0.45)，均分
    start = -0.45  # 起始坐标
    
    grid_points = []
    for i in range(m_prime):
        for j in range(m_prime):
            dx = start + j * step
            dy = start + i * step
            distance = np.sqrt(dx**2 + dy**2)  # 到中心的欧氏距离
            grid_points.append((distance, dx, dy))
    
    # 步骤4：按距离中心从近到远排序
    grid_points.sort(key=lambda x: x[0])
    
    # 步骤5：取前 n 个点，只返回 (dx, dy)
    offsets = [(point[1], point[2]) for point in grid_points[:n]]
    
    return offsets

# ---------- 辅助函数：计算价值热力图 ----------
def compute_heatmap(env, V):
    """
    计算价值热力图
    
    对于每个可行驶格子(包括起始格子),取所有以该点为 cur 的状态的 V 值的最大值。
    终点格子不计算热力值(保持为 NaN)。
    
    参数:
        env: InertialRacingEnv 实例
        V: numpy 数组,每个状态的价值
    
    返回:
        heatmap: numpy array, shape (h, w), 每个格子的最大 V 值
                 不可行驶格子为 -inf, 终点格子为 NaN
    """
    h, w = env.grid.shape
    heatmap = np.full((h, w), -np.inf)
    
    # 遍历所有状态,更新热力值
    for state_idx in range(env.num_states):
        cur, prev = env.idx_to_state[state_idx]
        x, y = cur
        
        # 跳过终点格子
        if (x, y) in env.goal_coords:
            continue
        
        # 更新热力值(取最大值)
        if V[state_idx] > heatmap[x, y]:
            heatmap[x, y] = V[state_idx]
    
    # 将终点格子设为 NaN(不渲染)
    for x, y in env.goal_coords:
        heatmap[x, y] = np.nan
    
    return heatmap


# ---------- 1. 赛道布局图 ----------
def draw_track(env, title="赛道布局", add_padding=True, save_to_file=False):
    """
    绘制赛道彩色图，四周可选灰色边框，自动显示终点奖励数字。
    
    参数:
        env: InertialRacingEnv 实例，需包含 grid 和 goal_rewards 属性
        title: 图片标题
        add_padding: 是否添加灰色边框
        save_to_file: True 保存到 outputs/ 文件夹，False 显示图片窗口
    """
    grid = env.grid
    goal_rewards = env.goal_rewards
    
    if add_padding:
        data = pad_with_gray(grid, pad_value=4)
        cmap = get_track_colormap()
        vmin, vmax = 0, 4
        offset = 1
    else:
        data = grid
        cmap = ListedColormap(['white', 'lime', 'red', 'black'])
        vmin, vmax = 0, 3
        offset = 0

    plt.figure(figsize=(6,6))
    plt.imshow(data, cmap=cmap, interpolation='none', vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.grid(False)
    plt.axis('off')

    # 显示终点奖励数字
    if goal_rewards is not None:
        h, w = grid.shape
        for i in range(h):
            for j in range(w):
                if grid[i, j] == 2:
                    reward = goal_rewards[i, j]
                    plt.text(j + offset, i + offset, f'{reward:.1f}',
                             ha='center', va='center',
                             color='white', fontsize=env.font_scale, weight='bold')
    
    if save_to_file:
        os.makedirs("outputs", exist_ok=True)
        safe_title = title.replace(" ", "_").replace("/", "_") + ".png"
        plt.savefig(os.path.join("outputs", safe_title), dpi=600, bbox_inches='tight')
        image_saved(f"图片已保存至: outputs/{safe_title}")
        plt.close()   # 关闭图形窗口，不显示
    else:
        plt.show()


# ---------- 2. 价值热力图 ----------
def draw_value_heatmap(env, V, title="价值热力图", add_padding=True, save_to_file=False):
    """
    绘制价值热力图,标记起始和终点格子
    
    参数:
        env: InertialRacingEnv 实例
        V: numpy 数组,每个状态的价值
        title: 图片标题
        add_padding: 是否添加灰色边框
        save_to_file: 是否保存图片(不显示)
    
    标记规则:
        - 起始格子: 绿色实线边框 + 绿色 "S" 字母(fontsize=env.font_scale * 3)
        - 终点格子: 红色实线边框 + 红色 "G" 字母(fontsize=env.font_scale * 3)
        - 普通格子: 仅显示热力图颜色
    """
    grid = env.grid
    h, w = grid.shape
    
    # 计算热力图
    heatmap = compute_heatmap(env, V)
    
    # 处理 padding
    if add_padding:
        data = pad_with_gray(grid, pad_value=4)
        cmap = get_track_colormap()
        vmin, vmax_cmap = 0, 4
        offset = 1
        
        # 热力图也添加 padding(NaN 边框)
        heatmap_padded = pad_heatmap_with_nan(heatmap)
    else:
        data = grid
        cmap = ListedColormap(['white', 'lime', 'red', 'black'])
        vmin, vmax_cmap = 0, 3
        offset = 0
        heatmap_padded = heatmap
    
    # 创建自定义 colormap: viridis + NaN 为灰色
    from matplotlib.colors import LinearSegmentedColormap
    viridis_cmap = plt.cm.viridis
    colors = viridis_cmap(np.linspace(0, 1, 256))
    # 设置 NaN 颜色为浅灰色
    colors_bad = np.array([0.8, 0.8, 0.8, 1.0])
    viridis_cmap.set_bad(color=colors_bad)
    
    # 绘制热力图
    plt.figure(figsize=(10, 10))
    im = plt.imshow(heatmap_padded, cmap=viridis_cmap, interpolation='nearest')
    plt.colorbar(im, label='Value')
    
    # 叠加赛道底图(半透明)
    plt.imshow(data, cmap=cmap, interpolation='none', vmin=vmin, vmax=vmax_cmap, alpha=0.3)
    
    plt.title(title)
    plt.grid(False)
    plt.axis('off')
    
    # 标记起始格子: 绿色实线边框 + 绿色 "S"
    for x, y in zip(*np.where(env.grid == 1)):
        rect = plt.Rectangle((y-0.5+offset, x-0.5+offset), 1, 1, 
                            fill=False, edgecolor='green', 
                            linewidth=3, linestyle='-')
        plt.gca().add_patch(rect)
        plt.text(y+offset, x+offset, 'S', ha='center', va='center',
                color='green', fontsize=env.font_scale * 3, weight='bold')
    
    # 标记终点格子: 红色实线边框 + 红色 "G"
    for x, y in zip(*np.where(env.grid == 2)):
        rect = plt.Rectangle((y-0.5+offset, x-0.5+offset), 1, 1, 
                            fill=False, edgecolor='red', 
                            linewidth=3, linestyle='-')
        plt.gca().add_patch(rect)
        plt.text(y+offset, x+offset, 'G', ha='center', va='center',
                color='red', fontsize=env.font_scale * 3, weight='bold')
    
    if save_to_file:
        os.makedirs("outputs", exist_ok=True)
        safe_title = title.replace(" ", "_").replace("/", "_") + ".png"
        plt.savefig(os.path.join("outputs", safe_title), dpi=600, bbox_inches='tight')
        image_saved(f"图片已保存至: outputs/{safe_title}")
        plt.close()
    else:
        plt.show()


# ---------- 2b. 价值热力图网格图（迭代过程可视化） ----------
def draw_value_heatmap_grid(env, V_slices, indices, title="价值迭代过程", 
                            add_padding=True, share_colorbar=True, save_to_file=False):
    """
    在2×3网格中绘制多个迭代轮次的价值热力图
    
    参数:
        env: InertialRacingEnv 实例
        V_slices: list of numpy arrays,采样的价值函数列表(通常6个)
        indices: list of int,对应的迭代轮数编号
        title: 总标题
        add_padding: 是否添加灰色边框
        share_colorbar: 是否共享色条
        save_to_file: 是否保存图片(不显示)
    
    注意:
        - 固定2行3列布局,最多6个子图
        - 每个子图标题显示迭代轮数: "Iteration {index}"
        - 如果切片数量少于6,多余子图自动隐藏
    """
    num_slices = len(V_slices)
    
    if num_slices == 0:
        print("警告：没有价值函数切片，无法绘制")
        return
    
    if share_colorbar:
        # 计算所有切片的热力图（使用原有的 compute_heatmap 函数）
        all_heatmaps = [compute_heatmap(env, V) for V in V_slices]
        all_values = []
        for hm in all_heatmaps:
            # 剔除 NaN 和 -inf（终点格子为 NaN，不可行驶格子为 -inf）
            finite_vals = hm[np.isfinite(hm)]
            if len(finite_vals) > 0:
                all_values.extend(finite_vals)
        if len(all_values) > 0:
            vmin_global = np.min(all_values)
            vmax_global = np.max(all_values)
        else:
            vmin_global, vmax_global = 0, 1  # fallback
    else:
        vmin_global, vmax_global = None, None
    
    # 创建2×3子图网格
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()  # 展平为1D数组方便索引
    
    grid = env.grid
    
    for idx, (V, iteration) in enumerate(zip(V_slices, indices)):
        ax = axes[idx]
        
        # 计算热力图
        heatmap = compute_heatmap(env, V)
        
        # 处理 padding
        if add_padding:
            data = pad_with_gray(grid, pad_value=4)
            cmap = get_track_colormap()
            vmin, vmax_cmap = 0, 4
            offset = 1
            
            # 热力图也添加 padding(NaN 边框)
            heatmap_padded = pad_heatmap_with_nan(heatmap)
        else:
            data = grid
            cmap = ListedColormap(['white', 'lime', 'red', 'black'])
            vmin, vmax_cmap = 0, 3
            offset = 0
            heatmap_padded = heatmap
        
        # 创建自定义 colormap: viridis + NaN 为灰色
        from matplotlib.colors import LinearSegmentedColormap
        viridis_cmap = plt.cm.viridis
        colors = viridis_cmap(np.linspace(0, 1, 256))
        # 设置 NaN 颜色为浅灰色
        colors_bad = np.array([0.8, 0.8, 0.8, 1.0])
        viridis_cmap.set_bad(color=colors_bad)
        
        # 绘制热力图
        im = ax.imshow(heatmap_padded, cmap=viridis_cmap, interpolation='nearest', vmin=vmin_global, vmax=vmax_global)
        
        # 叠加赛道底图(半透明)
        ax.imshow(data, cmap=cmap, interpolation='none', vmin=vmin, vmax=vmax_cmap, alpha=0.3)
        
        ax.set_title(f"Iteration {iteration}", fontsize=10)
        ax.grid(False)
        ax.axis('off')
        
        # 标记起始格子: 绿色实线边框 + 绿色 "S"
        for x, y in zip(*np.where(env.grid == 1)):
            rect = plt.Rectangle((y-0.5+offset, x-0.5+offset), 1, 1, 
                                fill=False, edgecolor='green', 
                                linewidth=3, linestyle='-')
            ax.add_patch(rect)
            ax.text(y+offset, x+offset, 'S', ha='center', va='center',
                    color='green', fontsize=env.font_scale * 3, weight='bold')
        
        # 标记终点格子: 红色实线边框 + 红色 "G"
        for x, y in zip(*np.where(env.grid == 2)):
            rect = plt.Rectangle((y-0.5+offset, x-0.5+offset), 1, 1, 
                                fill=False, edgecolor='red', 
                                linewidth=3, linestyle='-')
            ax.add_patch(rect)
            ax.text(y+offset, x+offset, 'G', ha='center', va='center',
                    color='red', fontsize=env.font_scale * 3, weight='bold')

    # 隐藏多余的子图
    for j in range(num_slices, len(axes)):
        axes[j].axis('off')
    
    if share_colorbar and vmin_global is not None and vmax_global is not None:
        # 调整子图布局，为 colorbar 留出右侧空间
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=vmin_global, vmax=vmax_global))
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, label='Value')
    
    # 设置总标题
    fig.suptitle(title, fontsize=14, y=1.02)
    
    if share_colorbar:
        plt.tight_layout(rect=[0, 0, 0.85, 1])
    else:
        plt.tight_layout()
    
    if save_to_file:
        os.makedirs("outputs", exist_ok=True)
        safe_title = title.replace(" ", "_").replace("/", "_") + ".png"
        plt.savefig(os.path.join("outputs", safe_title), dpi=600, bbox_inches='tight')
        image_saved(f"图片已保存至: outputs/{safe_title}")
        plt.close()
    else:
        plt.show()


# ---------- 2. 状态价值箭头图（已废弃，请使用 draw_value_heatmap） ----------
def _draw_state_arrows_deprecated(env, V, title="状态价值箭头图", add_padding=True, 
                      arrow_alpha=0.6, min_linewidth=0.5, max_linewidth=4.0,
                      save_to_file=False):
    """
    【已废弃】此函数已被 draw_value_heatmap 替代
    
    在赛道底图上绘制所有有限价值的状态转移箭头，箭头从 prev 指向 cur，线宽与价值 V(s) 成正比。
    
    参数:
        env: InertialRacingEnv 实例
        V: numpy 数组，每个状态的价值
        title: 图片标题
        add_padding: 是否添加灰色边框
        arrow_alpha: 箭头透明度 (0~1)
        min_linewidth, max_linewidth: 箭头线宽范围
        save_to_file: 是否保存图片（不显示）
    """
    grid = env.grid
    h, w = grid.shape
    
    # 根据地图尺寸动态调整线宽
    scale = env.visual_scale
    scaled_min_linewidth = min_linewidth * scale
    scaled_max_linewidth = max_linewidth * scale
    
    # 提取所有有效状态（价值有限，排除 -inf 的死亡状态）
    state_indices = [i for i in range(env.num_states) if np.isfinite(V[i])]
    if not state_indices:
        print("警告：没有有效状态价值，无法绘制箭头")
        return
    
    # 获取有效状态的价值数组，用于归一化线宽
    valid_V = V[state_indices]
    v_min, v_max = valid_V.min(), valid_V.max()
    if v_max - v_min < 1e-6:
        v_min, v_max = v_min - 0.5, v_min + 0.5  # 避免除零
    
    # 绘制底图（同 draw_track 但不显示终点数字）
    if add_padding:
        data = pad_with_gray(grid, pad_value=4)
        cmap = get_track_colormap()
        vmin, vmax_cmap = 0, 4
        offset = 1
    else:
        data = grid
        cmap = ListedColormap(['white', 'lime', 'red', 'black'])
        vmin, vmax_cmap = 0, 3
        offset = 0
    
    plt.figure(figsize=(8, 8))
    plt.imshow(data, cmap=cmap, interpolation='none', vmin=vmin, vmax=vmax_cmap)
    plt.title(title)
    plt.grid(False)
    plt.axis('off')
    
    # 绘制箭头
    from matplotlib.patches import FancyArrowPatch
    
    arrows = []
    for s_idx in state_indices:
        cur, prev = env.idx_to_state[s_idx]
        # 起点和终点坐标（已考虑 padding 偏移）
        start = (prev[1] + offset, prev[0] + offset)   # (x, y)
        end   = (cur[1] + offset, cur[0] + offset)
        value = V[s_idx]
        # 线性映射线宽
        width = scaled_min_linewidth + (value - v_min) / (v_max - v_min) * (scaled_max_linewidth - scaled_min_linewidth)
        
        # 跳过零长度箭头，改用虚线圆环标记自循环状态
        if start == end:
            from matplotlib.patches import Circle
            # 绘制蓝色虚线圆环表示自循环/停留状态
            circle = Circle(start, radius=0.15, 
                           fill=False, edgecolor='blue', 
                           linewidth=width, linestyle='--', alpha=arrow_alpha)
            plt.gca().add_patch(circle)
            continue

        arrow = FancyArrowPatch(start, end,
                                arrowstyle='->,head_length=4,head_width=3',
                                linewidth=width,
                                edgecolor='blue',
                                alpha=arrow_alpha,
                                zorder=10)
        arrows.append(arrow)
    
    for arrow in arrows:
        plt.gca().add_patch(arrow)
    
    if save_to_file:
        os.makedirs("outputs", exist_ok=True)
        safe_title = title.replace(" ", "_").replace("/", "_") + ".png"
        plt.savefig(os.path.join("outputs", safe_title), dpi=600, bbox_inches='tight')
        image_saved(f"图片已保存至: outputs/{safe_title}")
        plt.close()
    else:
        plt.show()


# ---------- 2b. 状态价值箭头网格图（迭代过程可视化，已废弃，请使用 draw_value_heatmap_grid） ----------
def _draw_state_arrows_grid_deprecated(env, V_slices, indices, title="价值迭代过程", 
                           add_padding=True, arrow_alpha=0.6, 
                           min_linewidth=0.5, max_linewidth=4.0,
                           save_to_file=False):
    """
    【已废弃】此函数已被 draw_value_heatmap_grid 替代
    
    在2×3网格中绘制多个迭代轮次的状态价值箭头图
    
    参数:
        env: InertialRacingEnv 实例
        V_slices: list of numpy arrays，采样的价值函数列表（通常6个）
        indices: list of int，对应的迭代轮数编号
        title: 总标题
        add_padding: 是否添加灰色边框
        arrow_alpha: 箭头透明度 (0~1)
        min_linewidth, max_linewidth: 箭头线宽范围
        save_to_file: 是否保存图片（不显示）
    
    注意:
        - 固定2行3列布局，最多6个子图
        - 每个子图标题显示迭代轮数："Iteration {index}"
        - 如果切片数量少于6，多余子图自动隐藏
        - 每个子图独立计算线宽归一化
    """
    num_slices = len(V_slices)
    
    if num_slices == 0:
        print("警告：没有价值函数切片，无法绘制")
        return
    
    # 根据地图尺寸动态调整线宽
    scale = env.visual_scale
    scaled_min_linewidth = min_linewidth * scale
    scaled_max_linewidth = max_linewidth * scale
    
    # 创建2×3子图网格
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()  # 展平为1D数组方便索引
    
    grid = env.grid
    
    for idx, (V, iteration) in enumerate(zip(V_slices, indices)):
        ax = axes[idx]
        
        # 提取有效状态
        state_indices = [i for i in range(env.num_states) if np.isfinite(V[i])]
        
        if not state_indices:
            # 绘制底图
            if add_padding:
                data = pad_with_gray(grid, pad_value=4)
                cmap = get_track_colormap()
                vmin, vmax_cmap = 0, 4
            else:
                data = grid
                cmap = ListedColormap(['white', 'lime', 'red', 'black'])
                vmin, vmax_cmap = 0, 3
            
            ax.imshow(data, cmap=cmap, interpolation='none', vmin=vmin, vmax=vmax_cmap)
            ax.set_title(f"Iteration {iteration}", fontsize=10)
            ax.grid(False)
            ax.axis('off')
            
            # 在合适位置标注
            ax.text(0.5, 0.5, "No valid states\n(all -inf)", 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='red', fontweight='bold')
            continue
        
        # 计算线宽归一化参数
        valid_V = V[state_indices]
        v_min, v_max = valid_V.min(), valid_V.max()
        if v_max - v_min < 1e-6:
            v_min, v_max = v_min - 0.5, v_min + 0.5
        
        # 绘制底图
        if add_padding:
            data = pad_with_gray(grid, pad_value=4)
            cmap = get_track_colormap()
            vmin, vmax_cmap = 0, 4
            offset = 1
        else:
            data = grid
            cmap = ListedColormap(['white', 'lime', 'red', 'black'])
            vmin, vmax_cmap = 0, 3
            offset = 0
        
        ax.imshow(data, cmap=cmap, interpolation='none', vmin=vmin, vmax=vmax_cmap)
        ax.set_title(f"Iteration {iteration}", fontsize=10)
        ax.grid(False)
        ax.axis('off')
        
        # 绘制箭头
        from matplotlib.patches import FancyArrowPatch
        
        for s_idx in state_indices:
            cur, prev = env.idx_to_state[s_idx]
            start = (prev[1] + offset, prev[0] + offset)
            end = (cur[1] + offset, cur[0] + offset)
            value = V[s_idx]
            width = min_linewidth + (value - v_min) / (v_max - v_min) * (max_linewidth - min_linewidth)
            
            # 跳过零长度箭头，改用虚线圆环标记自循环状态
            if start == end:
                from matplotlib.patches import Circle
                # 绘制蓝色虚线圆环表示自循环/停留状态
                circle = Circle(start, radius=0.15, 
                               fill=False, edgecolor='blue', 
                               linewidth=width, linestyle='--', alpha=arrow_alpha)
                ax.add_patch(circle)
                continue

            arrow = FancyArrowPatch(start, end,
                                    arrowstyle='->,head_length=4,head_width=3',
                                    linewidth=width,
                                    edgecolor='blue',
                                    alpha=arrow_alpha,
                                    zorder=10)
            ax.add_patch(arrow)
    
    # 隐藏多余的子图
    for j in range(num_slices, len(axes)):
        axes[j].axis('off')
    
    # 设置总标题
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_to_file:
        os.makedirs("outputs", exist_ok=True)
        safe_title = title.replace(" ", "_").replace("/", "_") + ".png"
        plt.savefig(os.path.join("outputs", safe_title), dpi=600, bbox_inches='tight')
        image_saved(f"图片已保存至: outputs/{safe_title}")
        plt.close()
    else:
        plt.show()


# ---------- 3. 最优轨迹图 ----------
def draw_trajectory(env, trajectory_data, title="最优轨迹", add_padding=True, 
                    arrow_linewidth=2.0, save_to_file=False):
    """
    绘制轨迹图（支持单条或多条轨迹）
    
    参数:
        env: InertialRacingEnv 实例
        trajectory_data: dict 或 list of dict
            - 单个轨迹字典（best_result格式）
            - 或多个轨迹列表（all_paths格式）
        title: 图片标题
        add_padding: 是否添加灰色边框
        arrow_linewidth: 箭头线宽
        save_to_file: 是否保存图片（不显示）
    
    注意:
        - 单条轨迹：纯黑色箭头
        - 多条轨迹：按回报从高到低排序，灰度渐变（黑→灰）
        - 自动排除失败路径（success=False）
        - 添加图例显示每条轨迹的"代价"（-total_reward）
        - 多轨迹时使用智能偏移避免重叠（高回报轨迹优先占据中心位置）
    """
    grid = env.grid
    
    # 根据地图尺寸动态调整线宽
    scale = env.visual_scale
    scaled_linewidth = arrow_linewidth * scale
    
    # 步骤1：统一输入格式
    if isinstance(trajectory_data, dict):
        trajectories = [trajectory_data]  # 转为单元素列表
    else:
        trajectories = trajectory_data
    
    # 步骤2：保留所有路径（包括失败的）
    all_paths = trajectories
    
    if not all_paths:
        print("警告：没有路径数据，无法绘制轨迹")
        return
    
    # 分离成功和失败路径
    success_paths = [t for t in all_paths if t.get('success', False)]
    failed_paths = [t for t in all_paths if not t.get('success', False)]
    
    # 成功路径按回报从高到低排序
    success_paths.sort(key=lambda x: x['total_reward'], reverse=True)
    
    if not success_paths and not failed_paths:
        print("警告：没有有效路径，无法绘制轨迹")
        return
    
    # 步骤3：生成灰度颜色映射（相同代价同色原则）
    n_success = len(success_paths)
    
    # 初始化 color_map，确保在所有分支中都可访问
    color_map = {}
    colors = []
    
    if n_success == 0:
        # 没有成功路径
        pass
    elif n_success == 1:
        # 单条轨迹用纯黑
        colors = ['#000000']
        cost = -success_paths[0]['total_reward']
        color_map[cost] = '#000000'
    else:
        # 收集所有成功轨迹的代价值
        costs = [-traj['total_reward'] for traj in success_paths]
        
        # 去重并排序得到唯一值列表
        unique_costs = sorted(set(costs))
        n_unique = len(unique_costs)
        
        # 根据唯一值数量分配颜色
        if n_unique == 1:
            # 只有一个唯一值，所有轨迹统一黑色
            color_map[unique_costs[0]] = '#000000'
        else:
            # 多个唯一值，按索引分配灰度
            # 第 i 个唯一值的灰度 = i / n_unique
            # 最后一个值是 (n_unique-1) / n_unique，避免纯白色
            for i, cost in enumerate(unique_costs):
                gray = i / n_unique  # 0.0 ~ (n_unique-1)/n_unique
                gray_int = int(gray * 255)
                color_map[cost] = f'#{gray_int:02x}{gray_int:02x}{gray_int:02x}'
        
        # 为每条轨迹分配颜色
        colors = [color_map[-traj['total_reward']] for traj in success_paths]
    
    # 失败路径统一用红色
    failed_color = '#FF0000'  # 纯红
    
    # 步骤4：生成智能偏移点（基于成功路径数量）
    offsets = _generate_sorted_offsets(n_success + len(failed_paths))
    
    # 步骤5：绘制底图（同 draw_track 但不显示终点数字）
    if add_padding:
        data = pad_with_gray(grid, pad_value=4)
        cmap = get_track_colormap()
        vmin, vmax = 0, 4
        offset = 1
    else:
        data = grid
        cmap = ListedColormap(['white', 'lime', 'red', 'black'])
        vmin, vmax = 0, 3
        offset = 0
    
    plt.figure(figsize=(8, 8))
    plt.imshow(data, cmap=cmap, interpolation='none', vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.grid(False)
    plt.axis('off')
    
    # 步骤6：绘制所有轨迹并收集图例句柄
    from matplotlib.patches import FancyArrowPatch
    import matplotlib.patches as mpatches
    
    legend_handles = []
    traj_idx = 0
    
    # 绘制成功路径（灰度渐变）
    for traj, color in zip(success_paths, colors):
        path = traj['path']
        
        # 获取该轨迹的偏移量
        dx, dy = offsets[traj_idx]
        traj_idx += 1
        
        # 绘制该轨迹的所有箭头
        for i in range(len(path) - 1):
            # path 中存储的是 (row, col) 格式，需要转换为绘图坐标 (x, y) 即 (col, row)
            # 应用偏移到每个坐标点
            start_pos = (path[i][1] + offset + dx, path[i][0] + offset + dy)
            end_pos = (path[i+1][1] + offset + dx, path[i+1][0] + offset + dy)
            
            # 跳过零长度箭头，改用虚线圆环标记自循环状态
            if start_pos == end_pos:
                from matplotlib.patches import Circle
                # 绘制虚线圆环表示自循环/停留状态
                circle = Circle(start_pos, radius=0.15, 
                               fill=False, edgecolor=color, 
                               linewidth=scaled_linewidth, linestyle='--', alpha=0.8)
                plt.gca().add_patch(circle)
                continue
            
            arrow = FancyArrowPatch(start_pos, end_pos,
                                    arrowstyle='->,head_length=4,head_width=3',
                                    linewidth=scaled_linewidth,
                                    edgecolor=color,
                                    alpha=0.8,
                                    zorder=10)
            plt.gca().add_patch(arrow)
    
    # 为成功路径创建图例句柄（相同代价只显示一行）
    if n_success > 0 and len(success_paths) > 0:
        # 收集所有唯一代价及其对应的颜色
        costs = [-traj['total_reward'] for traj in success_paths]
        unique_costs = sorted(set(costs))
        
        # 为每个唯一代价创建一个图例句柄
        for cost in unique_costs:
            color = color_map[cost]
            label = f"代价: {cost:.1f}"
            handle = mpatches.Patch(color=color, label=label)
            legend_handles.append(handle)
    
    # 绘制失败路径（红色虚线）
    for traj in failed_paths:
        path = traj['path']
        
        # 获取该轨迹的偏移量
        dx, dy = offsets[traj_idx]
        traj_idx += 1
        
        # 绘制该轨迹的所有箭头（红色虚线）
        for i in range(len(path) - 1):
            start_pos = (path[i][1] + offset + dx, path[i][0] + offset + dy)
            end_pos = (path[i+1][1] + offset + dx, path[i+1][0] + offset + dy)
            
            # 跳过零长度箭头，改用虚线圆环标记自循环状态
            if start_pos == end_pos:
                from matplotlib.patches import Circle
                # 绘制红色虚线圆环表示自循环/停留状态
                circle = Circle(start_pos, radius=0.15, 
                               fill=False, edgecolor=failed_color, 
                               linewidth=scaled_linewidth, linestyle='--', alpha=0.5)
                plt.gca().add_patch(circle)
                continue
            
            arrow = FancyArrowPatch(start_pos, end_pos,
                                    arrowstyle='->,head_length=4,head_width=3',
                                    linewidth=scaled_linewidth,
                                    edgecolor=failed_color,
                                    linestyle='--',  # 虚线
                                    alpha=0.5,
                                    zorder=10)
            plt.gca().add_patch(arrow)
    
    # 添加失败路径图例（如果有）
    if failed_paths:
        failed_handle = mpatches.Patch(color=failed_color, 
                                      label="Failed",
                                      linestyle='--')
        legend_handles.append(failed_handle)
    
    # 步骤7：添加图例
    if legend_handles:
        plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1), 
                   fontsize=8, title="轨迹性能", framealpha=0.9)
    
    # 步骤8：添加起点标记（仅多轨迹模式）
    if isinstance(trajectory_data, list):
        # 多轨迹模式：显示全部轨迹的起点标记
        for i, traj in enumerate(success_paths):
            path = traj['path']
            color = colors[i]
            cost = -traj['total_reward']  # 代价 = -总回报
            
            # 获取起点坐标 (row, col)
            start_row, start_col = path[0]
            
            # 转换为绘图坐标 (x, y) = (col + offset, row + offset)
            x = start_col + offset
            y = start_row + offset
            
            # 绘制实心圆圈（稍大但比网格格子小，半径约0.35-0.4）
            from matplotlib.patches import Circle
            circle_radius = 0.35
            circle = Circle((x, y), radius=circle_radius, 
                           facecolor=color, edgecolor='none',
                           alpha=0.7, zorder=15)
            plt.gca().add_patch(circle)
            
            # 在圆圈中心添加白色代价数字
            plt.text(x, y, f'{cost:.1f}',
                    ha='center', va='center',
                    color='white', fontsize=env.font_scale, weight='bold',
                    zorder=20)
    # 单轨迹模式（isinstance(trajectory_data, dict)）：不添加任何标记
    
    if save_to_file:
        os.makedirs("outputs", exist_ok=True)
        safe_title = title.replace(" ", "_").replace("/", "_") + ".png"
        plt.savefig(os.path.join("outputs", safe_title), dpi=600, bbox_inches='tight')
        image_saved(f"图片已保存至: outputs/{safe_title}")
        plt.close()
    else:
        plt.show()


# ---------- 3b. 轨迹网格图（策略演化可视化） ----------
def draw_trajectory_grid(env, best_paths, indices, title="策略演化轨迹", 
                        add_padding=True, arrow_linewidth=2.0,
                        save_to_file=False):
    """
    在网格中绘制多个策略切片的最优轨迹
    
    参数:
        env: InertialRacingEnv 实例
        best_paths: list of dict，最优路径列表
                   每个元素为 find_best_start_and_path 返回的 best_result
        indices: list of int，对应的迭代轮数编号
        title: 总标题
        add_padding: 是否添加灰色边框
        arrow_linewidth: 箭头线宽
        save_to_file: 是否保存图片（不显示）
    
    注意:
        - 行数规则：<=8个路径用2行，>=9个路径用3行
        - 列数自适应：根据路径数量自动计算
        - 每个子图标题显示迭代轮数："Iteration {index}"
        - 如果路径数量少于格子数，多余子图自动隐藏
        - 每个子图只有一条轨迹，无需偏移算法
        - 所有轨迹使用纯黑色箭头
    """
    num_paths = len(best_paths)
    
    if num_paths == 0:
        print("警告：没有路径数据，无法绘制")
        return
    
    # 根据路径数量确定行列数
    if num_paths <= 8:
        n_rows = 2
        n_cols = (num_paths + 1) // 2  # 向上取整
    else:
        n_rows = 3
        n_cols = (num_paths + 2) // 3  # 向上取整
    
    # 创建子图网格
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    
    # 处理单行或单列的情况，确保axes是二维数组
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    axes = axes.flatten()  # 展平为1D数组方便索引
    
    grid = env.grid
    
    # 根据地图尺寸动态调整线宽
    scale = env.visual_scale
    scaled_linewidth = arrow_linewidth * scale
    
    for idx, (path_dict, iteration) in enumerate(zip(best_paths, indices)):
        ax = axes[idx]
        
        # 处理完全没有路径的情况（理论上不应该发生）
        if path_dict is None:
            ax.text(0.5, 0.5, "No path found", ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f"Iteration {iteration}", fontsize=10)
            ax.axis('off')
            continue
        
        path = path_dict['path']
        is_success = path_dict.get('success', False)
        
        # 对于从一开始就原地不动的轨迹，需要重复其path到两次以被后续检测到
        if len(path) == 1 and path[0][0] == path[0][1]:
            path = path * 2
        
        # 根据成功状态决定颜色
        if is_success:
            arrow_color = 'black'
            circle_color = 'black'
            alpha = 0.8
        else:
            arrow_color = '#FF0000'  # 红色
            circle_color = '#FF0000'
            alpha = 0.6
        
        # 绘制底图
        if add_padding:
            data = pad_with_gray(grid, pad_value=4)
            cmap = get_track_colormap()
            vmin, vmax_cmap = 0, 4
            offset = 1
        else:
            data = grid
            cmap = ListedColormap(['white', 'lime', 'red', 'black'])
            vmin, vmax_cmap = 0, 3
            offset = 0
        
        ax.imshow(data, cmap=cmap, interpolation='none', vmin=vmin, vmax=vmax_cmap)
        ax.set_title(f"Iteration {iteration}", fontsize=10)
        ax.grid(False)
        ax.axis('off')
        
        # 绘制轨迹箭头
        from matplotlib.patches import FancyArrowPatch
        import matplotlib.patches as mpatches
        
        for i in range(len(path) - 1):
            # path 中存储的是 (row, col) 格式，需要转换为绘图坐标 (x, y) 即 (col, row)
            start_pos = (path[i][1] + offset, path[i][0] + offset)
            end_pos = (path[i+1][1] + offset, path[i+1][0] + offset)
            
            # 跳过零长度箭头，改用虚线圆环标记自循环状态
            if start_pos == end_pos:
                from matplotlib.patches import Circle
                # 绘制虚线圆环表示自循环/停留状态
                circle = Circle(start_pos, radius=0.15, 
                               fill=False, edgecolor=circle_color, 
                               linewidth=2.0, linestyle='--', alpha=alpha)
                ax.add_patch(circle)
                continue
            
            arrow = FancyArrowPatch(start_pos, end_pos,
                                    arrowstyle='->,head_length=4,head_width=3',
                                    linewidth=scaled_linewidth,
                                    edgecolor=arrow_color,
                                    alpha=alpha,
                                    zorder=10)
            ax.add_patch(arrow)
        
        # 添加图例
        legend_handles = []
        if is_success:
            cost_value = -path_dict['total_reward']
            label = f"代价: {cost_value:.1f}"
            handle = mpatches.Patch(color=arrow_color, label=label)
            legend_handles.append(handle)
        else:
            failed_handle = mpatches.Patch(color=arrow_color, 
                                          label="Failed",
                                          linestyle='--')
            legend_handles.append(failed_handle)
        
        if legend_handles:
            ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1), 
                     fontsize=8, title="轨迹性能", framealpha=0.9)
    
    # 隐藏多余的子图
    total_subplots = n_rows * n_cols
    for j in range(num_paths, total_subplots):
        axes[j].axis('off')
    
    # 设置总标题
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_to_file:
        os.makedirs("outputs", exist_ok=True)
        safe_title = title.replace(" ", "_").replace("/", "_") + ".png"
        plt.savefig(os.path.join("outputs", safe_title), dpi=600, bbox_inches='tight')
        image_saved(f"图片已保存至: outputs/{safe_title}")
        plt.close()
    else:
        plt.show()

def draw_q_learning_curve(episode_rewards, title="Q-learning 学习曲线",
                          save_to_file=False, window=None, max_points=100000):
    """
    绘制每幕总回报的曲线，并叠加滑动平均。

    参数:
        episode_rewards: list of float，每幕的总回报
        title:           图表标题
        save_to_file:    是否保存图片
        window:          滑动平均窗口大小
        max_points:      最大绘制的散点数量，超过此值时自动均匀抽样（默认100000）
    """
    len_episode_rewards = len(episode_rewards)
    
    # 处理窗口大小
    if window is None:
        window = max(1, len_episode_rewards // 10)
    else:
        window = min(max(1, window), len_episode_rewards)
    
    info(f"总幕数: {len_episode_rewards}，滑动平均窗口大小: {window}")

    # 转换为代价（负回报）
    episode_costs_original = [-reward if reward != -np.inf else np.inf for reward in episode_rewards]
    
    # 处理无穷大值：用失败代价替代
    finite_costs = [c for c in episode_costs_original if c != np.inf]
    if finite_costs:
        fail_reward = 1.1 * max(finite_costs)
    else:
        fail_reward = 1000.0  # 默认失败代价
    
    episode_costs = [fail_reward if c == np.inf else c for c in episode_costs_original]
    
    # 如果数据量超过阈值，进行均匀抽样
    if len_episode_rewards > max_points:
        warning(f"数据量过大 ({len_episode_rewards} > {max_points})，进行均匀抽样...")
        
        sample_indices = np.linspace(0, len_episode_rewards - 1, max_points, dtype=int)
        sample_indices = np.unique(sample_indices)
        
        sampled_costs = [episode_costs[i] for i in sample_indices]
        sampled_episodes = sample_indices.tolist()
        
        warning(f"抽样后点数: {len(sampled_costs)}")
        label_episode_cost = "Episode Cost (sampled)"
        
        costs_for_plotting = sampled_costs
        episodes_for_plotting = sampled_episodes
    else:
        label_episode_cost = "Episode Cost"
        costs_for_plotting = episode_costs
        episodes_for_plotting = list(range(len_episode_rewards))
    
    plt.figure(figsize=(8, 5))
    # 绘制失败代价参考线
    plt.axhline(y=fail_reward, color='black', linestyle='--', linewidth=0.5, label='Failure Cost', zorder=10)
    # 绘制散点图
    plt.scatter(episodes_for_plotting, costs_for_plotting, s=1, alpha=0.3, c='gray', label=label_episode_cost)
    
    # 使用 pandas 快速计算滑动中位数和四分位距
    costs_series = pd.Series(costs_for_plotting, dtype=float)
    
    # 中心滚动窗口，最少2个点才计算
    rolling = costs_series.rolling(window=window, center=True, min_periods=2)
    medians = rolling.median()
    lower   = rolling.quantile(0.25)
    upper   = rolling.quantile(0.75)
    
    # 处理窗口内只有1个点的情况：用原值填充（与原逻辑一致）
    medians = medians.fillna(costs_series)
    lower   = lower.fillna(medians)
    upper   = upper.fillna(medians)
    
    # 绘制曲线（x 轴用 0..N-1，与原代码行为一致）
    # x_range = range(len(medians))
    plt.plot(episodes_for_plotting, medians, linewidth=0.7, label=f'Median (w={window})')
    plt.fill_between(episodes_for_plotting, lower, upper, alpha=0.4, label='IQR 25-75%')
    
    plt.xlabel('Episode')
    plt.ylabel('Total Cost')
    plt.yscale('log')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_to_file:
        os.makedirs("outputs", exist_ok=True)
        safe_title = title.replace(" ", "_").replace("/", "_") + ".png"
        plt.savefig(os.path.join("outputs", safe_title), dpi=300, bbox_inches='tight')
        image_saved(f"图片已保存至: outputs/{safe_title}")
        plt.close()
    else:
        plt.show()

def compute_q_heatmap(env, Q):
    """
    根据 Q 表计算每个格子的热力值（格子作为 cur 的所有状态中 max_a Q(s,a) 的最大值）

    参数:
        env: InertialRacingEnv 实例
        Q: defaultdict 或 dict，键为 (cur, prev) 元组，值为长度 num_actions 的 np.array

    返回:
        heatmap: np.array, shape (H, W)，普通格子为数值，不可行驶格子为 -inf，终点格子为 NaN
    """
    h, w = env.grid.shape
    heatmap = np.full((h, w), -np.inf)

    # 遍历所有状态（直接从 env 的状态索引遍历，确保覆盖所有可达状态）
    for state_idx in range(env.num_states):
        cur, prev = env.idx_to_state[state_idx]
        x, y = cur

        # 跳过终点格子
        if (x, y) in env.goal_coords:
            continue

        # 获取该状态的 Q 值，如果没有则跳过（理论上应有）
        state = (cur, prev)
        if state not in Q:
            continue
        q_vals = Q[state]
        # 计算 V(s) = max_a Q(s,a)（剔除初始化为 0 的这些未被更新的值）
        if np.all(q_vals == 0):
            continue
        v = np.max(q_vals[q_vals < 0])
        if np.isfinite(v) and v > heatmap[x, y]:
            heatmap[x, y] = v

    # 终点格子设为 NaN
    for x, y in env.goal_coords:
        heatmap[x, y] = np.nan

    return heatmap

def draw_q_value_heatmap(env, Q, title="Q值热力图", add_padding=True,
                         vmin=None, vmax=None, save_to_file=False):
    """
    绘制 Q-learning 价值热力图

    参数:
        env: InertialRacingEnv 实例
        Q: Q 表（defaultdict or dict）
        title: 标题
        add_padding: 是否加灰色边框
        vmin, vmax: 颜色映射范围（若为 None 则自动计算）
        save_to_file: 是否保存
    """
    grid = env.grid
    heatmap = compute_q_heatmap(env, Q)

    # 处理 padding
    if add_padding:
        data = pad_with_gray(grid, pad_value=4)
        cmap = get_track_colormap()
        vmin_cmap, vmax_cmap = 0, 4
        offset = 1
        heatmap_padded = pad_heatmap_with_nan(heatmap)
    else:
        data = grid
        cmap = ListedColormap(['white', 'lime', 'red', 'black'])
        vmin_cmap, vmax_cmap = 0, 3
        offset = 0
        heatmap_padded = heatmap

    # 如果未指定 vmin/vmax，从当前热力图计算
    finite_vals = heatmap[np.isfinite(heatmap)]
    if vmin is None:
        vmin = np.min(finite_vals) if len(finite_vals) > 0 else -10
    if vmax is None:
        vmax = np.max(finite_vals) if len(finite_vals) > 0 else 0

    # 创建 colormap（NaN 为灰色）
    viridis_cmap = plt.cm.viridis
    viridis_cmap.set_bad(color='lightgray')

    plt.figure(figsize=(8, 8))
    im = plt.imshow(heatmap_padded, cmap=viridis_cmap, interpolation='nearest',
                    vmin=vmin, vmax=vmax)
    plt.colorbar(im, label='Value')
    plt.imshow(data, cmap=cmap, interpolation='none', vmin=vmin_cmap, vmax=vmax_cmap, alpha=0.3)
    plt.title(title)
    plt.axis('off')

    # 标记起点和终点（复用原逻辑）
    for x, y in zip(*np.where(env.grid == 1)):
        rect = plt.Rectangle((y-0.5+offset, x-0.5+offset), 1, 1,
                            fill=False, edgecolor='green', linewidth=3, linestyle='-')
        plt.gca().add_patch(rect)
        plt.text(y+offset, x+offset, 'S', ha='center', va='center',
                color='green', fontsize=env.font_scale*3, weight='bold')
    for x, y in zip(*np.where(env.grid == 2)):
        rect = plt.Rectangle((y-0.5+offset, x-0.5+offset), 1, 1,
                            fill=False, edgecolor='red', linewidth=3, linestyle='-')
        plt.gca().add_patch(rect)
        plt.text(y+offset, x+offset, 'G', ha='center', va='center',
                color='red', fontsize=env.font_scale*3, weight='bold')

    if save_to_file:
        os.makedirs("outputs", exist_ok=True)
        safe_title = title.replace(" ", "_").replace("/", "_") + ".png"
        plt.savefig(os.path.join("outputs", safe_title), dpi=600, bbox_inches='tight')
        image_saved(f"图片已保存至: outputs/{safe_title}")
        plt.close()
    else:
        plt.show()

def draw_q_value_heatmap_grid(env, Q_slices, indices, title="Q学习价值演化",
                              add_padding=True, share_colorbar=True,
                              save_to_file=False):
    """
    绘制多个 Q 表快照的热力图网格

    参数:
        env: InertialRacingEnv
        Q_slices: list of Q 表（每个元素为 defaultdict or dict）
        indices: list of int，对应的训练幕数编号
        title: 总标题
        add_padding: 是否加边框
        share_colorbar: 是否统一颜色范围（基于所有切片）
        save_to_file: 是否保存
    """
    num_slices = len(Q_slices)
    if num_slices == 0:
        print("警告：没有 Q 表切片")
        return

    # 计算每个切片的热力图
    heatmaps = [compute_q_heatmap(env, Q) for Q in Q_slices]

    # 确定全局颜色范围
    vmin_global, vmax_global = None, None
    if share_colorbar:
        all_vals = []
        for hm in heatmaps:
            finite = hm[np.isfinite(hm)]
            if len(finite) > 0:
                all_vals.extend(finite)
        if all_vals:
            vmin_global = np.min(all_vals)
            vmax_global = np.max(all_vals)

    # 确定子图布局（同轨迹网格规则）
    if num_slices <= 8:
        n_rows = 2
        n_cols = (num_slices + 1) // 2
    else:
        n_rows = 3
        n_cols = (num_slices + 2) // 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    axes = axes.flatten()

    grid = env.grid
    for idx, (hm, iteration) in enumerate(zip(heatmaps, indices)):
        ax = axes[idx]

        if add_padding:
            data = pad_with_gray(grid, pad_value=4)
            cmap = get_track_colormap()
            vmin_cmap, vmax_cmap = 0, 4
            offset = 1
            hm_padded = pad_heatmap_with_nan(hm)
        else:
            data = grid
            cmap = ListedColormap(['white', 'lime', 'red', 'black'])
            vmin_cmap, vmax_cmap = 0, 3
            offset = 0
            hm_padded = hm

        viridis_cmap = plt.cm.viridis
        viridis_cmap.set_bad(color='lightgray')
        im = ax.imshow(hm_padded, cmap=viridis_cmap, interpolation='nearest',
                       vmin=vmin_global, vmax=vmax_global)
        ax.imshow(data, cmap=cmap, interpolation='none', vmin=vmin_cmap, vmax=vmax_cmap, alpha=0.3)
        ax.set_title(f"Iteration {iteration}", fontsize=10)
        ax.axis('off')

        # 标记 S/G
        for x, y in zip(*np.where(env.grid == 1)):
            rect = plt.Rectangle((y-0.5+offset, x-0.5+offset), 1, 1,
                                fill=False, edgecolor='green', linewidth=2, linestyle='-')
            ax.add_patch(rect)
            ax.text(y+offset, x+offset, 'S', ha='center', va='center',
                    color='green', fontsize=env.font_scale*2.5, weight='bold')
        for x, y in zip(*np.where(env.grid == 2)):
            rect = plt.Rectangle((y-0.5+offset, x-0.5+offset), 1, 1,
                                fill=False, edgecolor='red', linewidth=2, linestyle='-')
            ax.add_patch(rect)
            ax.text(y+offset, x+offset, 'G', ha='center', va='center',
                    color='red', fontsize=env.font_scale*2.5, weight='bold')

    # 隐藏多余子图
    for j in range(num_slices, len(axes)):
        axes[j].axis('off')

    # 添加全局 colorbar
    if share_colorbar and vmin_global is not None and vmax_global is not None:
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=vmin_global, vmax=vmax_global))
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, label='Value')

    fig.suptitle(title, fontsize=14, y=1.02)

    if share_colorbar:
        plt.tight_layout(rect=[0, 0, 0.85, 1])
    else:
        plt.tight_layout()
    
    if save_to_file:
        os.makedirs("outputs", exist_ok=True)
        safe_title = title.replace(" ", "_").replace("/", "_") + ".png"
        plt.savefig(os.path.join("outputs", safe_title), dpi=600, bbox_inches='tight')
        image_saved(f"图片已保存至: outputs/{safe_title}")
        plt.close()
    else:
        plt.show()