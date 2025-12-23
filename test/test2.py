from openai import OpenAI
from openai.types.chat import ChatCompletionMessage
import os
from dotenv import load_dotenv

#load env
load_dotenv()

client = OpenAI(
            api_key=os.environ.get('DEEPSEEK_API_KEY'), 
            base_url="https://api.deepseek.com"
        )

response = client.chat.completions.create(
    model="deepseek-chat",
    messages = [{"role":"user",
                "content":"你好"}]
)

print(response.choices[0].message.content)