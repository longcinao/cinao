# -*- coding: utf-8 -*-
"""
控制台编码处理
====================================================
Windows 控制台默认 GBK，打印中文会抛 UnicodeEncodeError 或显示乱码。
这里把 stdout/stderr 统一切换为 UTF-8（与高光谱图像.py 的做法一致）。
"""

import sys


def setup_console():
    """把标准输出/错误流切到 UTF-8，失败时静默忽略。"""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
