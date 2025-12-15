from core.llm import LLM
from core.message import UserMessage , SystemMessage

def main():
    #初始化LLM
    llm = LLM(model="deepseek-chat")

    #对话历史
    history = [SystemMessage("你是一个暴躁的变成助手，喜欢用反问句回答问题"),
               UserMessage("你好，我想学习写一个Agent 框架")]
    
    print("🤖 正在思考...")

    response = llm.chat(history)

    print(f"User: {history[-1].content}")
    print(f"AI:   {response.content}")

if __name__ == "__main__":
    main()