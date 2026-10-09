# -*- coding: utf-8 -*-
"""工具子包。

刻意不在此处 import losses/metrics（两者依赖 torch），
这样 utils.console 可被树莓派上的 deploy/infer_tflite.py 单独导入，而无需安装 torch。
"""
