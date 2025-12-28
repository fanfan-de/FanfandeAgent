from core.llm.llm import LLM
from core.message.message import UserMessage , SystemMessage,LLMMessage,ToolMessage
from core.memory.simple_memory import SimpleMemory
from core.tools.tool_manager import ToolManager
from rich import print as rprint
import asyncio
import os
from pydantic import AnyUrl
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from mcp.shared.context import RequestContext
from mcp.shared.metadata_utils import get_display_name
from mcp.server.fastmcp import FastMCP
from core.tools.mcp.mcp_client import MCPClient 



async def run_agent_workflow():

    #初始化对话历史
    memory = SimpleMemory(system_message=SystemMessage(content = "你是一个暴躁的编程助手，喜欢用反问句回答问题"),
                    history=[])

    #初始化LLM
    llm = LLM(model="deepseek-chat")

    #mcp
    #创建MCp客户端
    mcp_client = MCPClient()
    await mcp_client.connect_to_server(".mcp_server.py")


    #user第一次的prompt输入
    user_input = input()
    memory.add(UserMessage(content=user_input))


    # 3. Agent 决策循环 (简易版)
    while True:
        #LLM 生成 response
        llmmessage = llm.chat(memory.to_messages())
        rprint(llmmessage)

        memory.add(llmmessage)
        
        # 如果 LLM 不需要调用工具，直接输出结果并停止
        if  llmmessage.tool_calls == None or llmmessage.tool_calls == []:
            print("\nAgent 回复:", llmmessage.content)

            #下一次输入
            user_input =input()
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

if __name__ == "__main__":
    # 使用 asyncio.run 启动顶层异步任务
    asyncio.run(run_agent_workflow())
                