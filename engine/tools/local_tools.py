from engine.agent import Agent
from engine.tools.tool import ToolRegistry




@agent.toolregistry.register
def add(a: int, b: int):
    """加法,将两个数相加"""
    return a + b