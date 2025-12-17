import os
from core.llm import LLM
from engine.agent import Agent
from engine.tool import ToolRegistry

# 初始化
llm = LLM() # 默认用 DeepSeek
# 故意设置一个很小的窗口，方便触发修剪
agent = Agent(llm=llm, system_prompt="你是一个数学家")
agent.memory.max_messages = 4 # 只保留最近 4 条，非常严苛



@agent.toolregistry.register
def add(a: int, b: int):
    """加法,将两个数相加"""
    return a + b

prompt="start"
while prompt != "end":
    prompt= input("user:")
    agent.run(prompt)


print("\n=== Debug: 查看当前记忆 ===")
for msg in agent.memory.to_messages():
    print(f"[{msg.role}] {msg.content} (ToolCalls: {getattr(msg, 'tool_calls', None)})")