from typing import Optional,Any,Dict
from pydantic import BaseModel

#定义消息基类
class Message(BaseModel):
    role: str
    #content 是一个变量
    #它的类型可以是 str（字符串）或 None
    #默认值为 None
    #Optional：来自 typing 模块，表示"可选的"
    content: Optional[str] = None # content 有可能是空的（比如只调用工具时）

    # 这一步是为了方便后续转成 OpenAI 需要的字典格式
    def to_deepseek_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content
        }

    

#kwargs 是关键字参数包，接收所有未明确声明的命名参数，以字典形式传递给函数，用于扩展参数处理的灵活性。

#定义不同角色的消息类
#system
class SystemMessage(Message):
    def __init__(self,content:str,**kwargs):
        super().__init__(role="system",content=content,**kwargs)

    def to_deepseek_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content
        }


#user
class UserMessage(Message):
    def __init__(self,content:Optional[str],**kwargs):
        super().__init__(role="user",content=content,**kwargs)

    def to_deepseek_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content
    }

#tools
class ToolMessage(Message):
    def __init__(self,content:Optional[str],**kwargs):
        super().__init__(role="tools",content=content,**kwargs)

#llm
class LLMMessage(Message):
    def __init__(self,content:Optional[str],**kwargs):
        super().__init__(role="llm",content=content,**kwargs)



#继承 `BaseModel`：** Pydantic 会自动帮我们检查类型。