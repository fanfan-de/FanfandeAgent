from core.llm.llm import LLM
from core.message.message import UserMessage , SystemMessage,LLMMessage,ToolMessage
from core.memory.simple_memory import SimpleMemory
from core.tools.tool_manager import ToolManager
from rich import print as rprint

def main():
    
    #对话历史
    memory = SimpleMemory(system_message=SystemMessage(content = "你是一个暴躁的编程助手，喜欢用反问句回答问题"),
                    history=[])
    #初始化LLM
    llm = LLM(model="deepseek-chat")

    #初始化工具服务
    # mcp_manager = MCPManager("tools/mcp/config.json")
    # mcp_manager.start_servers()
    # llm_tools = mcp_manager.get_combined_tools()
    unified_tool_registry = ToolManager("tools/mcp/config.json")



    # 3. Agent 决策循环 (简易版)
    while True:

        user_input =  input()
        memory.add(UserMessage(content=user_input))

        native_response = llm.chat(memory.to_messages())
        print(native_response)
        response = native_response.choices[0].message
        memory.add(LLMMessage(content=response.content,tool_calls=response.tool_calls))
        # 如果 LLM 不需要调用工具，直接输出结果并停止
        if response.tool_calls == None:
            print("\nAgent 回复:", response.content)
            break

        # 4. 如果 LLM 需要调用工具
        memory.add(LLMMessage(content=response.content,tool_calls=response.tool_calls))
        memory.append({"role": "assistant", "content": response.content})
        
        for content_block in response.content:
            if content_block.type == "tool_use":
                tool_name = content_block.name
                tool_args = content_block.input
                tool_use_id = content_block.id

                # 执行 MCP 工具
                result = ToolManager.handle_tool_call(tool_name, tool_args)
                
                # 将结果反馈给 LLM (处理 result 可能包含 text 或 image 的情况)
                tool_result_content = []
                for item in result:
                    if hasattr(item, 'text'):
                        tool_result_content.append({"type": "text", "text": item.text})

                memory.Add(
                    ToolMessage(content=tool_result_content,
                                 tool_use_id=tool_use_id))
                #     "role": "user",
                #     "content": [
                #         {
                #             "type": "tool_result",
                #             "tool_use_id": tool_use_id,
                #             "content": tool_result_content,
                #         }
                #     ],
                # })
                print(f"工具执行完毕，正在发回 LLM...")





    



if __name__ == "__main__":
    main()