from core.llm.llm import LLM
from core.message.message import UserMessage , SystemMessage,LLMMessage,ToolMessage
from core.memory.simple_memory import SimpleMemory
from core.tools.tool_manager import ToolManager
from rich import print as rprint
import json  # 确保文件头部导入了 json
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
    memory = SimpleMemory(system_message=SystemMessage(content = "你是一个有求必以的助手"),
                    history=[])

    #初始化LLM
    llm = LLM(model="deepseek-reasoner")

    #mcp
    #创建MCp客户端
    mcp_client = MCPClient()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 拼接成完整路径 (E:\Project\FanfandeAgent\.mcp_server.py)
    server_path = os.path.join(base_dir, "core/tools/mcp/mcp_server.py")
    print(f"正在连接 Server: {server_path}") # 打印出来看路径对不对！
    await mcp_client.connect_to_server(server_path)

    tools =  await mcp_client.session.list_tools()
    available_tools = [{ 
            "type": "function",  #必须有这个字段
            "function":{
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in tools.tools]
    


    #user第一次的prompt输入
    user_input = input()
    memory.add(UserMessage(content=user_input))


    # 3. Agent 决策循环 (简易版)
    while True:
        #LLM 生成 response
        llmmessage = llm.chat(memory.to_messages(),tools=available_tools)
        rprint(llmmessage)

        memory.add(llmmessage)
        
        print("\n--------开始解析LLM Response----------\n")
        # 如果 LLM 不需要调用工具，直接输出结果并停止
        if  llmmessage.tool_calls == None or llmmessage.tool_calls == []:

            print("\nAgent 回复:", llmmessage.content)

            #下一次输入
            user_input =input()
            memory.add(UserMessage(content=user_input))
            continue


        print("\n--------需要调用tools----------\n")   
        # 4. 如果 LLM 需要调用工具
        for toolcall in  llmmessage.tool_calls:
            if toolcall.type == "function":
                result = await mcp_client.session.call_tool(name=toolcall.function.name,arguments=json.loads(toolcall.function.arguments))
                rprint(result)
                memory.add(ToolMessage(content= result.content[0].text, tool_call_id=toolcall.id))   
        
        print(f"\n-----------------工具执行完毕,存入memory----------------------\n")



if __name__ == "__main__":
    # 使用 asyncio.run 启动顶层异步任务
    asyncio.run(run_agent_workflow())
                