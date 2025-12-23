from . import custom_tools_manager

@custom_tools_manager.register.register
def add(a: int, b: int):
    """加法,将两个数相加"""
    return a + b

@custom_tools_manager.register
def get_system_time():
    """获取系统当前时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")