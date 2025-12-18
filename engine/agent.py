# 需要导入新的 Memory 和 Message 类
from core.memory import Memory
from core.message import UserMessage, SystemMessage, ToolMessage,LLMMessage
from engine.tool import function_to_schema ,ToolRegistry
from openai.types.chat import ChatCompletionMessageFunctionToolCall
from openai.types.chat import ChatCompletionMessageToolCall
from typing import List
import json
from rich import print as rprint
# ... 其他原有导入


class Agent:
    def __init__(self, llm, system_prompt: str = "你是一个有用的 AI 助手。"):
        self.llm = llm
        #初始化工具模块
        self.toolregistry = ToolRegistry()
        # 初始化记忆模块
        self.memory = Memory(
            system_message=SystemMessage(content = system_prompt),
            max_messages=15 # 设定一个合理的窗口值
        )

    
    
    # _execute_tool 方法保持不变...
    def _execute_tool(self,tool_call:ChatCompletionMessageFunctionToolCall)->dict:
        fun = self.toolregistry.get_tool(tool_call.function.name)
        func_args = json.loads(tool_call.function.arguments)
        result = fun(**func_args)
        #print("result=" + str(result))
        tool_result = dict()
        tool_result["tool_call_id"] = tool_call.id
        tool_result["content"] = str(result)
        tool_result["role"] = "tool"
        tool_result["name"] = tool_call.function.name
        return  tool_result


    def run(self, prompt: str, max_turns: int = 5):
        # 1. 将用户输入存入记忆
        self.memory.add(UserMessage(content = prompt))
        current_turn = 0
        while current_turn < max_turns:
            current_turn += 1

            messages_to_send = self.memory.to_messages()
            rprint( messages_to_send)
            response_message = self.llm.chat(messages_to_send, self.toolregistry.tool_schemas)

            #存LLMMessage 到memory


            # 5. 判断是否调用工具
            if response_message.tool_calls:
                message = LLMMessage(content = response_message.content,
                tool_calls=LLMMessage.To_ToolCalls(response_message.tool_calls))
                self.memory.add(message)
                #self.memory.add(temp)
                for tool_call in response_message.tool_calls:
                    # 执行工具
                    tool_result_dict = self._execute_tool(tool_call)
                    #print("tool_result_dict:")
                    #print(tool_result_dict)
                    # 封装为 ToolMessage 对象
                    tool_msg = ToolMessage(
                        content=tool_result_dict["content"],
                        tool_call_id=tool_result_dict["tool_call_id"]
                    )
                    #print("tool_msg:" + str(tool_msg))
                    
                    # 将工具结果存入记忆
                    self.memory.add(tool_msg)
                
                # 循环继续，下一轮 LLM 会看到工具结果
            else:
                message = LLMMessage(content = response_message.content)
                self.memory.add(message)
                print(f"🤖 Answer: {response_message.content}")
                return response_message.content
        
        return "Max turns reached."