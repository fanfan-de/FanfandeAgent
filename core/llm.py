import os
from dotenv import load_dotenv
from typing import  List, Optional, Dict, Any, Union
from core.message import Message, LLMMessage
from deepseek import DeepSeekAPI
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage
from rich import print as rprint
from rich.console import Console


#load env
load_dotenv()

class LLM:
    #init
    def __init__(self,model:str="deepseek-chat"):
        self.model = model
        self.client = OpenAI(
            api_key=os.environ.get('DEEPSEEK_API_KEY'), 
            base_url="https://api.deepseek.com"
        )

    def chat(self,messages:List[Message],tools:Optional[List[Dict]]=None)->Union[LLMMessage,ChatCompletionMessage]:
        """
        核心方法：发送消息列表，获取 AI 回复
        :param messages: 历史消息列表
        :param tools: (新增) 工具的 JSON Schema 列表
        """
        try:
            ## 2. 构造 API 参数
            params = {
                "model": self.model,
                "messages": [msg.model_dump() for msg in messages],
                "stream": False,
            }

             # 3. 如果有工具，则传入 tools 参数
            if tools:
                params["tools"] = tools
                params["tool_choice"] = "auto" # 让模型自动决定是否调用

            # 发起请求
            response = self.client.chat.completions.create(**params)

            # 获取完整的 Message 对象 (包含 content 和 tool_calls)
            response_message = response.choices[0].message

            # 4. 关键决策点：
            # 如果模型返回了 tool_calls，我们需要把这个原始对象给回去，
            # 否则 Agent 无法解析参数。
            # 如果只是普通对话，我们可以继续用你的 LLMMessage 封装。

            if response_message.tool_calls:
            # 情况 A: 模型想要调用工具 -> 返回原始对象 (或者你需要升级 LLMMessage 来存储 tool_calls)
                return response_message

            else:
                # 情况 B: 普通文本回复 -> 保持你原有的封装习惯
                return LLMMessage(content=response_message.content,tool_calls = None)
            
        except Exception as e:
            print(f"调用 LLM 出错: {e}")
            # 打印最后一条消息以便调试
            if messages:
                print(f"Last Msg: {messages[-1]}")
            return LLMMessage(content="对不起，我遇到了一点错误。")

        # #核心方法：发送消息列表，获取 AI 回复
        # client = self.client
        # try:
        #     response = client.chat.completions.create(
        #         model=self.model,
        #         messages = [msg.to_deepseek_dict() for msg in messages],
        #         stream=False
        #     )  
        #     response_message = response.choices[0].message
        #     content = response_message.content

        #     return LLMMessage(content=content)
        
        # except Exception as e:
        #     print(f"调用 LLM 出错: {e}")
        #     print(messages)
        #     return LLMMessage(content="对不起，我遇到了一点错误。")

