import asyncio
import json
import os
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPManager:
    def __init__(self, config_path: str = "tools/mcp/config.json"):
        self.config_path = config_path
        self.sessions = {}  # server_name -> session
        self.exit_stack = AsyncExitStack()
        self.tool_to_server = {} # tool_name -> server_name (用于路由)

    async def start_servers(self):
        """根据配置文件启动所有 MCP Server"""
        with open(self.config_path, 'r') as f:
            config = json.load(f)

        for server_name, server_config in config.get("mcpServers", {}).items():
            print(f"正在启动 MCP Server: {server_name}...")
            
            # 配置启动参数
            params = StdioServerParameters(
                command=server_config["command"],
                args=server_config["args"],
                env={**os.environ, **server_config.get("env", {})}
            )

            # 建立 stdio 连接
            # 使用 AsyncExitStack 确保程序退出时自动关闭子进程
            transport = await self.exit_stack.enter_async_context(stdio_client(params))
            read, write = transport
            
            # 建立 Session
            session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            
            self.sessions[server_name] = session
            print(f"Server {server_name} 已连接")

    async def get_combined_tools(self):
        """获取所有 Server 的工具并转换为 LLM 格式"""
        all_llm_tools = []
        for server_name, session in self.sessions.items():
            result = await session.list_tools()
            for tool in result.tools:
                # 为了防止工具重名，我们在内部记录映射关系
                # 如果你想更保险，可以在 tool.name 前加 server_name 前缀
                self.tool_to_server[tool.name] = server_name
                
                all_llm_tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                })
        return all_llm_tools

    async def call_tool(self, tool_name: str, arguments: dict):
        """根据工具名，找到对应的 Server 并执行"""
        server_name = self.tool_to_server.get(tool_name)
        if not server_name:
            raise ValueError(f"未知工具: {tool_name}")
        
        session = self.sessions[server_name]
        print(f"正在调用 {server_name} 的工具: {tool_name}...")
        result = await session.call_tool(tool_name, arguments)
        return result.content

    async def stop_all(self):
        """关闭所有连接"""
        await self.exit_stack.aclose()