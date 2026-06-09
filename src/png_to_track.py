# src/png_to_track.py
# 将 PNG 图像（4 色索引）转换为赛道文本文件
"""
用法：
    python src/png_to_track.py input.png output.txt

也可在其他脚本中导入使用：
    from src.png_to_track import png_to_track
    png_to_track('my_track.png', 'tracks/my_track.txt')
"""

from PIL import Image
import numpy as np
import sys

# 默认颜色 → 字符映射
DEFAULT_COLOR_MAP = {
    (255, 255, 255): 'O',   # 白色 → 普通道路
    (0, 255, 0):     'S',   # 绿色 → 起点
    (255, 0, 0):     'E',   # 红色 → 终点
    (0, 0, 0):       'X',   # 黑色 → 障碍物
}


def png_to_track(png_path, txt_path, color_map=None):
    """
    将 PNG 图像转换为赛道文本文件。

    参数:
        png_path:   输入 PNG 文件路径
        txt_path:   输出 txt 文件路径
        color_map:  可选，字典，将 RGB 元组映射到字符 'O','S','E','X'
                    默认使用 DEFAULT_COLOR_MAP
    """
    if color_map is None:
        color_map = DEFAULT_COLOR_MAP

    img = Image.open(png_path).convert('RGB')
    arr = np.array(img)
    h, w = arr.shape[:2]

    lines = []
    for i in range(h):
        row_chars = []
        for j in range(w):
            pixel = tuple(arr[i, j])
            ch = color_map.get(pixel, 'O')   # 未知颜色默认普通道路
            row_chars.append(ch)
        lines.append(''.join(row_chars))

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"赛道已保存至: {txt_path}  ({h}x{w})")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("用法: python src/png_to_track.py <input.png> <output.txt>")
        sys.exit(1)
    png_to_track(sys.argv[1], sys.argv[2])
