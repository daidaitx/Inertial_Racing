# src/environment.py
import numpy as np
from collections import deque
from src.utils import success, info

# ---------- 1. 辅助函数：Bresenham 直线算法 ----------
def get_line_points(x0, y0, x1, y1):
    """
    返回从 (x0,y0) 到 (x1,y1) 直线经过的所有整数格点（包含起点？不包含终点？根据需要调整）。
    这里我们希望检查从 cur 到 next_cur 的所有中间格子（包括 next_cur 但不包括 cur），
    所以返回包含终点但不包含起点的点列表。
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    err = dx - dy
    x, y = x0, y0
    while (x, y) != (x1, y1):
        # 注意：先移动再记录，或者先记录再移动？我们要的是从起点出发经过的格子（不含起点）
        # 这里先移动一步再记录，确保不包含起点
        if x == x0 and y == y0:
            pass  # 跳过起点
        else:
            points.append((x, y))
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    points.append((x1, y1))  # 包含终点
    return points

def is_path_clear(grid, cur, next_cur):
    """
    检查从 cur 到 next_cur 的直线路径上所有格子是否均可行驶（障碍物为 3）。
    cur 和 next_cur 都是 (x,y) 坐标。
    """
    x0, y0 = cur
    x1, y1 = next_cur
    points = get_line_points(x0, y0, x1, y1)
    for (x, y) in points:
        if not (0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]):
            return False
        if grid[x, y] == 3:   # 障碍
            return False
    return True

# ---------- 新增：无模型交互的转移计算 ----------
def compute_raw_transition(cur, prev, action, grid, drivable_set, goal_coords, goal_rewards):
    """
    根据物理规则计算从 (cur, prev) 执行动作 action 的结果。

    参数:
        cur, prev:    (x,y) 坐标
        action:       (dx,dy) 动作向量
        grid:         整数网格 (H,W)
        drivable_set: 可行驶格子集合
        goal_coords:  终点格子集合
        goal_rewards: 终点奖励矩阵 (H,W)

    返回:
        (type, data)
        type: 'death' -> data=None
              'goal'  -> data=reward (float)
              'normal' -> data=(next_cur, cur)   # 注意：返回的 next_state
    """
    vx = cur[0] - prev[0]
    vy = cur[1] - prev[1]
    temp = (cur[0] + vx, cur[1] + vy)
    next_cur = (temp[0] + action[0], temp[1] + action[1])

    # 检查 next_cur 是否在可行驶集合中
    if next_cur not in drivable_set:
        return ('death', None)

    # 穿墙检测
    if not is_path_clear(grid, cur, next_cur):
        return ('death', None)

    # 到达终点
    if next_cur in goal_coords:
        reward = goal_rewards[next_cur[0], next_cur[1]]
        return ('goal', reward)

    # 正常移动
    return ('normal', (next_cur, cur))

# ---------- 2. 环境主类 ----------
class InertialRacingEnv:
    def __init__(self, grid):
        """
        参数:
            grid: (H,W) int, 0=普通,1=起点,2=终点,3=障碍
        """
        self.grid = grid
        self.H, self.W = grid.shape
        
        # 计算可视化缩放因子（基于地图最大维度，平方根反比）
        self.max_dim = max(self.H, self.W)
        self.visual_scale = np.sqrt(10.0 / self.max_dim)  # 以10为基准尺度
        self.font_scale = min(10.0, 10.0 * 10.0 / self.max_dim)  # 以10pt为基准字体大小
        
        # 自动计算终点奖励矩阵
        self.goal_rewards = self._compute_goal_rewards()
        
        # 可行驶格子集合（值 0,1,2）
        self.drivable = np.where((grid == 0) | (grid == 1) | (grid == 2))
        self.drivable_coords = list(zip(self.drivable[0], self.drivable[1]))
        self.drivable_set = set(self.drivable_coords)
        self.num_drivable = len(self.drivable_coords)
        
        # 起点格子列表
        start_coords = list(zip(*np.where(grid == 1)))
        self.start_states = [(c, c) for c in start_coords]  # (cur, prev) 初始速度为0
        
        # 终点区域格子集
        self.goal_coords = set(zip(*np.where(grid == 2)))
        
        # 动作列表
        self.actions = [(dx, dy) for dx in (-1,0,1) for dy in (-1,0,1)]
        self.num_actions = len(self.actions)
        
        # 以下属性将在 BFS 后填充
        self.state_to_idx = {}   # 将 (cur, prev) 映射到整数 idx
        self.idx_to_state = []   # idx -> (cur, prev)
        self.transitions = []    # transitions[idx][a] = (type, data)
                                 # type: 0=normal -> data = next_idx
                                 #       1=goal   -> data = goal_reward
                                 #       2=death  -> data = None
        self.num_states = 0
        
        # 执行 BFS 构建状态空间和转移表
        self._build_state_space()
        
        # 输出环境创建完成信息
        success(f"✓ 环境创建完成!")
        info(f"  可行驶格子数: {self.num_drivable}")
        info(f"  理论状态数: {self.num_drivable ** 2} ({self.num_drivable}²)")
        info(f"  BFS合法状态数: {self.num_states}")
        compression_rate = (1 - self.num_states / (self.num_drivable ** 2)) * 100
        info(f"  状态空间压缩率: {compression_rate:.2f}%")
    
    def _compute_goal_rewards(self):
        """内部方法：计算每个终点格子的深入奖励 (0~10)"""
        grid = self.grid
        h, w = grid.shape
        is_goal = (grid == 2)
        goal_positions = list(zip(*np.where(is_goal)))
        if not goal_positions:
            return np.zeros((h, w), dtype=float)
        
        from collections import deque
        distance = np.full((h, w), -1, dtype=int)
        queue = deque()
        dirs = [(1,0), (-1,0), (0,1), (0,-1)]
        
        # 找表层终点（与可行驶非终点相邻）
        for x, y in goal_positions:
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w:
                    if grid[nx, ny] in (0, 1):
                        distance[x, y] = 0
                        queue.append((x, y))
                        break
        
        # BFS 向内传播
        while queue:
            x, y = queue.popleft()
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and is_goal[nx, ny] and distance[nx, ny] == -1:
                    distance[nx, ny] = distance[x, y] + 1
                    queue.append((nx, ny))
        
        max_dist = np.max(distance[is_goal]) if np.any(is_goal) else 0
        if max_dist == 0:
            return np.zeros((h, w), dtype=float)
        
        goal_reward = np.zeros((h, w), dtype=float)
        for x, y in goal_positions:
            d = distance[x, y]
            goal_reward[x, y] = 10.0 * d / max_dist
        return goal_reward
    
    def _build_state_space(self):
        """BFS 从所有初始状态出发，收集所有可达状态，并预计算转移"""
        queue = deque()
        # 初始状态入队
        for s in self.start_states:
            if s not in self.state_to_idx:
                idx = self.num_states
                self.state_to_idx[s] = idx
                self.idx_to_state.append(s)
                self.num_states += 1
                queue.append(s)
        
        trans_dict = {}
        
        while queue:
            cur_state = queue.popleft()
            cur, prev = cur_state
            
            # 对于每个动作
            for a in self.actions:
                # 计算转移（复用独立函数）
                typ, data = compute_raw_transition(
                    cur, prev, a, self.grid, self.drivable_set, self.goal_coords, self.goal_rewards
                )

                if typ == 'death':
                    trans_dict.setdefault(cur_state, {})[a] = ('death', None)
                elif typ == 'goal':
                    trans_dict.setdefault(cur_state, {})[a] = ('goal', data)
                else:  # normal
                    next_state = data  # 就是 (next_cur, cur)
                    # 如果该状态未在状态集中，则加入并放入队列
                    if next_state not in self.state_to_idx:
                        new_idx = self.num_states
                        self.state_to_idx[next_state] = new_idx
                        self.idx_to_state.append(next_state)
                        self.num_states += 1
                        queue.append(next_state)
                    # 记录转移
                    trans_dict.setdefault(cur_state, {})[a] = ('normal', self.state_to_idx[next_state])
        
        # 将转移字典转换为列表形式，便于价值迭代快速访问
        self.transitions = [None] * self.num_states
        for state, action_map in trans_dict.items():
            sidx = self.state_to_idx[state]
            # 初始化该状态的动作结果列表，长度=num_actions，默认死亡
            action_results = [('death', None) for _ in range(self.num_actions)]
            for a_idx, a in enumerate(self.actions):
                if a in action_map:
                    action_results[a_idx] = action_map[a]
            self.transitions[sidx] = action_results
        
        # 确保所有状态都有转移记录（没有动作记录的状态默认全死亡，不可能出现，但安全起见）
        for i in range(self.num_states):
            if self.transitions[i] is None:
                self.transitions[i] = [('death', None) for _ in range(self.num_actions)]
    
    def get_num_states(self):
        return self.num_states
    
    def get_num_actions(self):
        return self.num_actions
    
    def get_start_state_indices(self):
        """返回所有初始状态对应的索引列表"""
        idxs = []
        for s in self.start_states:
            if s in self.state_to_idx:
                idxs.append(self.state_to_idx[s])
        return idxs
    
    def get_transition(self, state_idx, action_idx):
        """返回 (type, data) 其中 type 为 'normal','goal','death'"""
        return self.transitions[state_idx][action_idx]
    
    def get_state_info(self, state_idx):
        """返回状态对应的 (cur, prev) 坐标"""
        return self.idx_to_state[state_idx]
    
    def step_raw(self, cur, prev, action_idx):
        """
        无模型交互接口：给定当前状态 (cur, prev) 和动作索引，返回 (next_state, reward, done)。

        参数:
            cur, prev:   坐标元组
            action_idx:  动作索引 (0~8)

        返回:
            next_state:  (next_cur, cur) 或 None（死亡/终点时）
            reward:      float，步惩罚-10 + 可能的终点奖励
            done:        bool，是否终止（死亡或到达终点）
        """
        a = self.actions[action_idx]
        typ, data = compute_raw_transition(
            cur, prev, a, self.grid, self.drivable_set, self.goal_coords, self.goal_rewards
        )

        if typ == 'death':
            return None, -np.inf, True
        elif typ == 'goal':
            reward = -10.0 + data
            return None, reward, True
        else:  # normal
            next_state = data   # (next_cur, cur)
            return next_state, -10.0, False
