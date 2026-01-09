from core.llm.llm import LLM
from core.message.message import UserMessage , SystemMessage,LLMMessage,ToolMessage
from core.memory.simple_memory import SimpleMemory
from rich import print as rprint
import json  
import asyncio
from core.tools.mcp.mcp_manager import MCPManager

async def run_agent_workflow():

    #初始化对话历史
    memory = SimpleMemory(system_message=SystemMessage(content = "你是一个助手，做事严谨认真，完成任务不要做多余的事，判断无法完成返回原因即可"),
                    history=[])

    #初始化LLM
    llm = LLM(model="deepseek-reasoner")

    #mcp
    #创建MCP客户端

    mcp_manager = MCPManager()
    await mcp_manager.start_servers()

    #rprint(await mcp_manager.get_combined_tools())

    tools =  await mcp_manager.get_combined_tools()

    for tool in tools:
        rprint(tool)
    available_tools = [{ 
        "type": "function",  #必须有这个字段
        "function":{
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["input_schema"]
        }
    } for tool in tools]


    #user第一次的prompt输入
    user_input = input()
    memory.add(UserMessage(content=user_input))


    # 3. Agent 决策循环 (简易版)
    while True:
        #LLM 生成 response
        rprint(memory)
        llmmessage = llm.chat(memory.to_messages(),tools=available_tools)
        #rprint(llmmessage)

        memory.add(llmmessage)
        
        print("\n--------开始解析LLM Response----------\n")
        # 如果 LLM 不需要调用工具，直接输出结果并停止
        if  llmmessage.tool_calls == None or llmmessage.tool_calls == []:

            #print("\nAgent 回复:", llmmessage.content)

            #下一次输入
            user_input =input()
            memory.add(UserMessage(content=user_input))
            continue


        print("\n--------需要调用tools----------\n")   
        # 4. 如果 LLM 需要调用工具
        for toolcall in  llmmessage.tool_calls:
            if toolcall.type == "function":
                result = await mcp_manager.call_tool(tool_name=toolcall.function.name,arguments=json.loads(toolcall.function.arguments))
                #rprint(result)
                memory.add(ToolMessage(content= result.content[0].text, tool_call_id=toolcall.id))   
        
        print(f"\n-----------------工具执行完毕,存入memory----------------------\n")



if __name__ == "__main__":
    # 使用 asyncio.run 启动顶层异步任务
    asyncio.run(run_agent_workflow())
                