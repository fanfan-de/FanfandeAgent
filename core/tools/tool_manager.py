from .custom.custom_tools_manager import CustomToolRegistry
from .mcp.mcp_manager import MCPManager


# class ToolManager:
#     def __init__(self):
#         self.custom

#     def add_tool(self, tool):
#         self.tools.append(tool)

#     def remove_tool(self, tool):
#         if tool in self.tools:
#             self.tools.remove(tool)

#     def get_tools(self):
#         return self.tools
    

class ToolManager:
    def __init__(self, mcp_config_path: str):
        self.mcp_manager = MCPManager(mcp_config_path)
        self.local_tools = {t["name"]: t for t in CustomToolRegistry.tool_schemas}
        self.all_tool_definitions = []

    async def initialize(self):
        """初始化：启动 MCP 并合并工具定义"""
        # 1. 启动 MCP Servers
        await self.mcp_manager.start_servers()
        
        # 2. 获取 MCP 工具定义
        mcp_tools = await self.mcp_manager.get_combined_tools()
        
        # 3. 合并本地工具定义（去掉 handler 引用，只保留 LLM 需要的 schema）
        local_definitions = []
        for name, info in self.local_tools.items():
            local_definitions.append({
                "name": info["name"],
                "description": info["description"],
                "input_schema": info["input_schema"]
            })
            
        self.all_tool_definitions = mcp_tools + local_definitions
        return self.all_tool_definitions

    async def handle_tool_call(self, tool_name: str, arguments: dict):
        """统一的执行入口"""
        # 情况 A: 如果是本地工具
        if tool_name in self.local_tools:
            print(f"[Local] 正在执行本地工具: {tool_name}")
            handler = self.local_tools[tool_name]["handler"]
            # 异步化调用本地函数（如果函数本身是同步的，可以用 run_in_executor，这里简化处理）
            result = handler(**arguments)
            return [{"type": "text", "text": str(result)}]

        # 情况 B: 如果是 MCP 工具
        else:
            print(f"[MCP] 正在分发至 MCP Server: {tool_name}")
            return await self.mcp_manager.call_tool(tool_name, arguments)

    async def shutdown(self):
        await self.mcp_manager.stop_all()

