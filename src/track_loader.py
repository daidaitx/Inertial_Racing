# src/track_loader.py
import numpy as np
from collections import deque
from src.utils import info, success

def load_track(txt_path):
    """从txt读取赛道，返回grid (H,W) 整型矩阵， 0=普通,1=起点,2=终点,3=障碍"""
    info(f"正在加载赛道文件: {txt_path}")
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    h = len(lines)
    w = len(lines[0]) if h > 0 else 0
    grid = np.zeros((h, w), dtype=int)
    char_map = {'O':0, 'S':1, 'E':2, 'X':3}
    for i, line in enumerate(lines):
        for j, ch in enumerate(line):
            if ch.upper() not in char_map:
                raise ValueError(f"Invalid char '{ch}' at ({i},{j})")
            grid[i, j] = char_map[ch.upper()]
    
    success(f"✓ 赛道加载成功! 尺寸: {h} x {w} (高 x 宽)")
    return grid
