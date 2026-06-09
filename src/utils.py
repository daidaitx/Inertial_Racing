# src/utils.py
"""
终端彩色输出工具模块

提供带颜色的信息输出函数,用于增强命令行界面的可读性。
支持 ANSI 转义码,适用于 Windows 10+、Linux、macOS 终端。
"""

import atexit
import os
from datetime import datetime
import time

# 全局日志文件句柄
_log_file = None

# ANSI 颜色代码定义
ORANGE = '\033[38;5;208m'   # 橘色(统一标签颜色)
GREEN = '\033[38;5;46m'     # 霓虹绿(成功信息,更醒目)
DARK_GREEN = '\033[38;5;22m' # 森林绿(图片保存成功,明显更深)
BLUE = '\033[94m'           # 蓝色(普通信息/进度)
YELLOW = '\033[93m'         # 黄色(警告/提示)
PINK = '\033[95m'           # 粉色(重要信息)
RESET = '\033[0m'           # 重置颜色


def setup_logging(log_dir="logs"):
    """初始化日志系统，创建日志文件并注册关闭函数"""
    global _log_file
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"run_{timestamp}.log")
    _log_file = open(log_path, "w", encoding="utf-8")
    atexit.register(close_logging)
    # 写入启动标记（可选）
    info(f"日志文件已创建: {log_path}")

def close_logging():
    """关闭日志文件句柄"""
    global _log_file
    if _log_file:
        _log_file.close()
        _log_file = None

def _write_log(msg):
    """内部函数：写入日志文件（纯文本，不带颜色代码）"""
    if _log_file:
        _log_file.write(msg + "\n")
        _log_file.flush()

def _get_timestamped_prefix():
    """
    生成带时间戳的前缀标签
    
    返回:
        str: 格式为 [Inertial_Racing YYYY-MM-DD HH:MM:SS] 的彩色前缀
    """
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    return f"{ORANGE}[Inertial_Racing {timestamp}]{RESET} "

def _get_plain_prefix():
    return f"[Inertial_Racing {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"

def success(msg):
    """
    输出亮绿色成功信息
    
    参数:
        msg: 要输出的消息字符串
    """
    # 控制台输出（带颜色）
    colored_msg = f"{_get_timestamped_prefix()}{GREEN}{msg}{RESET}"
    print(colored_msg)
    # 日志输出（纯文本）
    plain_msg = f"{_get_plain_prefix()} {msg}"
    _write_log(plain_msg)


def image_saved(msg):
    """
    输出深绿色图片保存成功信息
    
    参数:
        msg: 要输出的消息字符串
    """
    # 控制台输出（带颜色）
    colored_msg = f"{_get_timestamped_prefix()}{DARK_GREEN}{msg}{RESET}"
    print(colored_msg)
    # 日志输出（纯文本）
    plain_msg = f"{_get_plain_prefix()} {msg}"
    _write_log(plain_msg)


def info(msg):
    """
    输出蓝色普通信息或进度信息
    
    参数:
        msg: 要输出的消息字符串
    """
    # 控制台输出（带颜色）
    colored_msg = f"{_get_timestamped_prefix()}{BLUE}{msg}{RESET}"
    print(colored_msg)
    # 日志输出（纯文本）
    plain_msg = f"{_get_plain_prefix()} {msg}"
    _write_log(plain_msg)


def warning(msg):
    """
    输出黄色警告信息
    
    参数:
        msg: 要输出的消息字符串
    """
    # 控制台输出（带颜色）
    colored_msg = f"{_get_timestamped_prefix()}{YELLOW}{msg}{RESET}"
    print(colored_msg)
    # 日志输出（纯文本）
    plain_msg = f"{_get_plain_prefix()} {msg}"
    _write_log(plain_msg)


def important(msg):
    """
    输出粉色重要信息
    
    参数:
        msg: 要输出的消息字符串
    """
    # 控制台输出（带颜色）
    colored_msg = f"{_get_timestamped_prefix()}{PINK}{msg}{RESET}"
    print(colored_msg)
    # 日志输出（纯文本）
    plain_msg = f"{_get_plain_prefix()} {msg}"
    _write_log(plain_msg)

class Timer:
    """
    简易计时器上下文管理器
    
    用法:
        with Timer("任务名称"):
            # 执行代码
            pass
    
    会自动在开始时输出开始时间，结束时输出耗时。
    """
    def __init__(self, task_name="任务"):
        self.task_name = task_name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        important(f"⏱️  开始: {self.task_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        
        # 格式化耗时
        if elapsed < 1:
            time_str = f"{elapsed * 1000:.2f} ms"
        elif elapsed < 60:
            time_str = f"{elapsed:.2f} s"
        else:
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            time_str = f"{minutes} min {seconds:.2f} s"
        
        important(f"✅ 完成: {self.task_name} | 耗时: {time_str}")
