from typing import Optional,Any,Dict
from pydantic import BaseModel

#定义消息基类
class Message(BaseModel):
    role: str
    content: Optional[str] = None


    

#kwargs 是关键字参数包，接收所有未明确声明的命名参数，以字典形式传递给函数，用于扩展参数处理的灵活性。

#定义不同角色的消息类
#system
class SystemMessage(Message):
    def __init__(self,content:str,**kwargs):
        super().__init__(role="system",content=content,**kwargs)



#user
class UserMessage(Message):
    def __init__(self,content:Optional[str],**kwargs):
        super().__init__(role="user",content=content,**kwargs)


#tools
# core/message.py

class ToolMessage(Message):
    tool_call_id: str
    
    def __init__(self, content: str, tool_call_id: str):
        # ⭐ 关键：将 tool_call_id 也传给父类的 __init__
        super().__init__(role="tool", content=content, tool_call_id=tool_call_id)
    

#llm
class LLMMessage(Message):
    tool_calls:Optional[str] = None


    def __init__(self,content:str,tool_calls:str):
        super().__init__(role="assistant",content=content,tool_calls=tool_calls)



