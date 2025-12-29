import os
from dotenv import load_dotenv
from typing import  List, Optional, Dict, Any, Union
from core.message.message import Message, LLMMessage
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
            print("\n发给LLM的内容\n-------------------------------------------------\n")
            rprint(params)
            print("\n")
            print("\n发起请求\n")
            response = self.client.chat.completions.create(**params)
            print("\n流式传输中.................................\n")
            #  处理流式响应
            full_response_content:str = ""
            #full_response_toolcall:str = ""
            #tool_calls:List[LLMMessage.ToolCall] = []
            # 1. 初始化累加器
            final_tool_calls = {}

            for chunk in response:
                # 提取流式增量内容
                content = chunk.choices[0].delta.content
                if content:
                    rprint(content, end="", flush=True)
                    full_response_content += content 

                deltatoolcalls = chunk.choices[0].delta.tool_calls
                rprint(deltatoolcalls)
                if deltatoolcalls:
                    for tool_call_delta in deltatoolcalls:
                        index = tool_call_delta.index
            #                 # 如果是该索引的第一个片段，初始化数据结构

                        if index not in final_tool_calls:
                            final_tool_calls[index] = LLMMessage.ToolCall(
                                id=tool_call_delta.id,
                                type = tool_call_delta.type,
                                function = LLMMessage.function(arguments=tool_call_delta.function.arguments, name=tool_call_delta.function.name)
                            )
                        
                        # 累加参数片段 (arguments 往往是分多次流式传输的)
                        if tool_call_delta.function.arguments:
                            final_tool_calls[index].function.arguments += tool_call_delta.function.arguments
                        
                        # 有时 id 或 name 也会在后续片段中才完整，虽然通常在第一个片段
                        if tool_call_delta.id:
                            final_tool_calls[index].id = tool_call_delta.id
                        if tool_call_delta.function.name:
                            final_tool_calls[index].function.name = tool_call_delta.function.name
                    
                    
            #         if toolcall[0].id!=None:
            #                 tool_calls.append(LLMMessage.ToolCall(id = toolcall[0].id,
            #                                             type=toolcall[0].type,function=toolcall[0].function))
            #         else:
            #                 tool_calls[-1].function.arguments += toolcall[0].function.arguments
                                

            #                 #tool_calls.append(chunk.choices[0].delta.tool_calls) if chunk.choices[0].delta.tool_calls else None

            # rprint(full_response_content + "full_response_content" )  # 换行
            # rprint(full_response_toolcall + "full_response_toocall" )  # 换行
            # # message in，llmmessage out
            return LLMMessage(content=full_response_content,tool_calls= list(final_tool_calls.values()) if final_tool_calls else None)
        
        except Exception as e:
            print(f"调用 LLM 出错: {e}")
            # 打印最后一条消息以便调试
            if messages:
                rprint(f"Last Msg: {messages}")
            return LLMMessage(content="对不起，我遇到了一点错误。")

