import os
from dotenv import load_dotenv
from typing import  List, Optional, Dict, Any, Union
from core.message.message import Message, LLMMessage
from deepseek import DeepSeekAPI
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage
from rich import print as rprint


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
    #输入输出
    def chat(self,messages:List[Message],tools:Optional[List[Dict]]=None)->Union[LLMMessage,ChatCompletionMessage]:
        """
        核心方法：发送消息列表，获取 AI 回复
        :param messages: 历史消息列表
        :param tools: (新增) 工具的 JSON Schema 列表
        """
        rprint(messages)
        try:
            ## 2. 构造 API 参数
            params = {
                "model": self.model,
                "messages": [msg.to_openai_dict() for msg in messages],
                #"messages": messages.model_dump(exclude_none=True),
                "stream": True,
            }

             # 3. 如果有工具，则传入 tools 参数
            if tools:
                params["tools"] = tools
                params["tool_choice"] = "auto" # 让模型自动决定是否调用

            rprint(params)
            # 发起请求
            response = self.client.chat.completions.create(**params)

            #  处理流式响应
            full_response_content:str = ""
            tool_calls:List[LLMMessage.ToolCall] = []

            for chunk in response:
                    # 提取流式增量内容
                        content = chunk.choices[0].delta.content
                        if content:
                            print(content, end="", flush=True)

                            full_response_content += content
                            tool_calls.append(chunk.choices[0].delta.tool_calls) if chunk.choices[0].delta.tool_calls else None

            # message in，llmmessage out
            return LLMMessage(content=full_response_content,tool_calls=tool_calls)
        
        except Exception as e:
            print(f"调用 LLM 出错: {e}")
            # 打印最后一条消息以便调试
            if messages:
                rprint(f"Last Msg: {messages}")
            return LLMMessage(content="对不起，我遇到了一点错误。")

