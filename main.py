from core.llm import LLM
from core.message import UserMessage , SystemMessage
from engine.tools.mcp import MCPManager

def main():
    text = input("please input:")
    #初始化LLM
    llm = LLM(model="deepseek-chat")

    #初始化工具服务
    mcp_manager = MCPManager("tools/mcp/config.json")
    mcp_manager.start_servers()
    llm_tools = mcp_manager.get_combined_tools()

    # 2. 初始化对话历史 (示例)
    messages = [{"role": "user", "content": "帮我读取桌面上的 test.txt 文件，并总结内容。"}]

    # 3. Agent 决策循环 (简易版)
    while True:

        response = llm.chat(messages)
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            tools=llm_tools,
            messages=messages
        )

        # 如果 LLM 不需要调用工具，直接输出结果并停止
        if response.stop_reason != "tool_use":
            print("\nAgent 回复:", response.content[0].text)
            break

        # 4. 如果 LLM 需要调用工具
        messages.append({"role": "assistant", "content": response.content})
        
        for content_block in response.content:
            if content_block.type == "tool_use":
                tool_name = content_block.name
                tool_args = content_block.input
                tool_use_id = content_block.id

                # 执行 MCP 工具
                result = await mcp_manager.call_tool(tool_name, tool_args)
                
                # 将结果反馈给 LLM (处理 result 可能包含 text 或 image 的情况)
                tool_result_content = []
                for item in result:
                    if hasattr(item, 'text'):
                        tool_result_content.append({"type": "text", "text": item.text})

                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": tool_result_content,
                        }
                    ],
                })
                print(f"工具执行完毕，正在发回 LLM...")




    #对话历史
    history = [SystemMessage(content="你是一个暴躁的变成助手，喜欢用反问句回答问题"),
               UserMessage(content= "你好，我想学习写一个Agent 框架")]
    

    #启动MCP服务器
    
    print("🤖 正在思考...")



    print(f"User: {history[-1].content}")
    print(f"AI:   {response.content}")

if __name__ == "__main__":
    main()