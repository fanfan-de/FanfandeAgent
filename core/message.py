from typing import Optional,Any,Dict
from typing import Optional, Any, List, Literal
from pydantic import BaseModel
from pydantic import Field
from openai.types.chat import ChatCompletionMessageFunctionToolCall

#定义消息基类
class Message(BaseModel):
    role: str
    content: Optional[str] = None

    def to_openai_dict(self):
        return self.model_dump()
    

    

#kwargs 是关键字参数包，接收所有未明确声明的命名参数，以字典形式传递给函数，用于扩展参数处理的灵活性。

#定义不同角色的消息类
#system
class SystemMessage(Message):
    role: Literal["system"] = "system"




#user
class UserMessage(Message):
    role: Literal["user"] = "user"


#tools
# core/message.py

class ToolMessage(Message):
    tool_call_id: str
    role: Literal["tool"] = "tool"
    

#llm
class LLMMessage(Message):
    #嵌套的basemodel结构
    class function(BaseModel):
        arguments:str
        name:str
    class ToolCall(BaseModel):
        id: str
        type: str = "function"
        function: "LLMMessage.function"

    tool_calls : Optional[list[ToolCall]]=None
    role: Literal["assistant"] = "assistant"

    def to_openai_dict(self):
        return {"role" : self.role,
                "content": self.content,
                "tool_calls":[e.model_dump() for e in self.tool_calls ] if self.tool_calls else None
        }
    
    @staticmethod
    def To_ToolCalls(ChatCompletionMessageFunctionToolCalls: List[Any]) ->Optional[List["LLMMessage.ToolCall"]]:
        """
        将 OpenAI SDK 返回的原生 ToolCall 对象列表转换为自定义的 Pydantic ToolCall 列表
        """
        if not ChatCompletionMessageFunctionToolCalls:
            return None
        
        res = []
        for tc in ChatCompletionMessageFunctionToolCalls:
            # 1. 构造内部的 function 对象
            # tc.function 包含 name 和 arguments
            f_obj = LLMMessage.function(
                name=tc.function.name,
                arguments=tc.function.arguments
            )
            
            # 2. 构造 ToolCall 对象
            # tc 包含 id, type 和 function
            tc_obj = LLMMessage.ToolCall(
                id=tc.id,
                type=tc.type,
                function=f_obj
            )
            res.append(tc_obj)
            
        return res if res else None


