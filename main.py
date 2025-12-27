from core.llm.llm import LLM
from core.message.message import UserMessage , SystemMessage,LLMMessage,ToolMessage
from core.memory.simple_memory import SimpleMemory
from core.tools.tool_manager import ToolManager
from rich import print as rprint

#初始化对话历史
memory = SimpleMemory(system_message=SystemMessage(content = "你是一个暴躁的编程助手，喜欢用反问句回答问题"),
                history=[])

#user第一次的prompt输入
user_input =  input()
memory.add(UserMessage(content=user_input))


# 3. Agent 决策循环 (简易版)
while True:
    #LLM 生成回复 
    llmmessage = llm.chat(memory.to_messages())
    
    print(llmmessage)

    memory.add(llmmessage)
    
    # 如果 LLM 不需要调用工具，直接输出结果并停止
    if  llmmessage.tool_calls == None:
        print("\nAgent 回复:", llmmessage.content)
        #下一次输入
        user_input =  input()
        memory.add(UserMessage(content=user_input))
        continue



    # 4. 如果 LLM 需要调用工具
    for toolcall in  llmmessage.tool_calls:
        if toolcall.type == "function":
            # tool_name = content_block.name
            # tool_args = content_block.input
            # tool_use_id = content_block.id

            # 执行 MCP 工具
            # result = ToolManager.handle_tool_call(tool_name, tool_args)
            
            # # 将结果反馈给 LLM (处理 result 可能包含 text 或 image 的情况)
            # tool_result_content = []
            # for item in result:
            #     if hasattr(item, 'text'):
            #         tool_result_content.append({"type": "text", "text": item.text})

            # memory.Add(
            #     ToolMessage(content=tool_result_content,
            #                     tool_use_id=tool_use_id))
            print(f"工具执行完毕，正在发回 LLM...")