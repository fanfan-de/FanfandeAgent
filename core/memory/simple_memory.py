from typing import List
from core.message.message import Message, SystemMessage
'''
    上下文窗口数量限制
    简单的message的叠加
'''

class SimpleMemory:
    def __init__(self,system_message: SystemMessage = None, history:List[Message]=[],max_messages: int = 10):
        """
        :param system_message: 系统提示词（永久保留）
        :param max_messages: 滑动窗口大小（保留最近的多少条消息）
        """
        self.system_message = system_message
        self.history: List[Message] = history # 存储 User, Assistant, Tool 消息
        self.max_messages = max_messages

    def add(self, message: Message):
        self.history.append(message)

    def to_messages(self) -> List[Message]:
        """
        生成发送给 LLM 的最终消息列表（包含修剪逻辑）
        """
        # 1. 简单的切片：取最后 N 条
        window_messages = self.history[-self.max_messages:]
        
        # 2. 安全检查（核心逻辑）：
        # 如果切片后的第一条消息是 'tool' (工具结果)，说明它对应的 'assistant' (工具调用) 被切掉了。
        # 这种情况发给 LLM 会报错，所以必须把这个孤儿 tool 消息也扔掉。
        # 我们一直扔，直到第一条不是 tool 为止。
        while window_messages and window_messages[0].role == 'tool':
            window_messages.pop(0)

        # 3. 组装最终列表
        final_messages = []
        if self.system_message:
            final_messages.append(self.system_message)
        
        final_messages.extend(window_messages)
        return final_messages
    
    def clear(self):
        self.history = []


