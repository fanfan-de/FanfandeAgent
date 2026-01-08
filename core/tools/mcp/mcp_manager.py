import asyncio
import json
import os
import sys
import shutil
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from rich import print as rprint
base_dir = os.path.dirname(os.path.abspath(__file__))
print(base_dir)

class MCPManager:
    #读取mcp 配置文件
    def __init__(self, config_path: str = base_dir + "\\config.json"):
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
            


            command = server_config["command"]
            args = server_config["args"]
            env = {**os.environ, **server_config.get("env", {})}

            # ---------------------------------------------------------
            # 【关键修复】Windows 平台兼容性处理
            # ---------------------------------------------------------
            if sys.platform == "win32":
                # 1. 解决 npx/npm 在 Windows 下需要 .cmd 后缀的问题
                if command in ["npx", "npm"] and not command.endswith(".cmd"):
                    command = f"{command}.cmd"
                
                # 2. 强制设置编码，防止中文乱码导致管道崩溃
                env["PYTHONIOENCODING"] = "utf-8"
            
            # 3. 检查命令是否存在 (可选，但推荐)
            if not shutil.which(command):
                print(f"[错误] 找不到命令: {command}。请检查是否安装了 Node.js 或相关依赖。")
                continue
            # ---------------------------------------------------------
            # 配置启动参数
            '''
            MCP 客户端Client用来告诉系统
            如何启动一个 MCP 服务器Server的配置清单

            
            '''
            params = StdioServerParameters(
                command=command, 
                args=args, 
                env=env
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
        return result

    async def stop_all(self):
        """关闭所有连接"""
        await self.exit_stack.aclose()

async def main():
    mcp_manager = MCPManager()
    
    try:
        # 1. 启动服务
        await mcp_manager.start_servers()
        
        # 2. 打印工具列表（测试连接成功）
        print("\n--- 获取工具列表 ---")
        tools = await mcp_manager.get_combined_tools()
        rprint(tools)

        # 3. 模拟保持运行（实际项目中这里是 Agent 的主循环）
        # 如果你不加等待，程序瞬间结束就会报错
        print("\n服务已启动。按 Ctrl+C 退出...")
        
        # 这是一个简单的永久等待，模拟服务器挂起
        # 实际开发中，这里可能是 while True: user_input = input(...)
        await asyncio.Event().wait() 

    except KeyboardInterrupt:
        print("\n用户中断，正在停止...")
    
    finally:
        # 4. 关键：确保退出时清理资源
        # 这会触发 AsyncExitStack 的退出，正确关闭 stdio_client
        print("正在关闭所有连接...")
        await mcp_manager.stop_all()
        print("所有连接已关闭。")

if __name__ == "__main__":
    # Windows 下有时需要设置策略以避免 ProactorEventLoop 的某些管道问题
    # 但通常上面的 try...finally 就能解决你的报错
    asyncio.run(main())