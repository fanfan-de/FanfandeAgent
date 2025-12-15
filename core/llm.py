import os
from dotenv import load_dotenv
from typing import List , Union
from core.message import Message, LLMMessage
from deepseek import DeepSeekAPI
from openai import OpenAI



#load env
load_dotenv()

class LLM:
    #init
    def __init__(self,model:str="deepseek-chat"):
        self.model = model
        self.client = DeepSeekAPI(
            api_key=os.getenv("DEEPSEEK_API_KEY")
            )

    def chat(self,messages:List[Message])->LLMMessage:
        #核心方法：发送消息列表，获取 AI 回复
        client = OpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages = [messages.to_deepseek_dict() for msg in messages],
                stream=False
            )  
            response_message = response.choices[0].message
            content = response_message.content

            return LLMMessage(content=content)
        
        except Exception as e:
            print(f"调用 LLM 出错: {e}")
            print(messages)
            return LLMMessage(content="对不起，我遇到了一点错误。")

