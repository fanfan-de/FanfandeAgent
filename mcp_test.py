import asyncio
import os
import traceback  # 引入堆栈打印工具
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run_github_agent():
    # 1. 配置 GitHub Token
    # 生产环境下建议从 .env 文件或环境变量读取
    github_token = "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"

    # 2. 配置 GitHub MCP Server 参数
    # 使用 npx 直接运行官方的 github server
    server_params = StdioServerParameters(
        command="npx.cmd",
        args=["--registry", "https://registry.npmmirror.com", "-y", "@modelcontextprotocol/server-github"],
        env={
            # 2. 检查环境变量：确保没有引入 os.environ 以排除系统旧 Token 的干扰
            "GITHUB_PERSONAL_ACCESS_TOKEN": github_token,
            "PATH": os.environ.get("PATH", "") # 必须保留 PATH 否则找不到 npx
        }
    )

    print("🚀 正在启动 GitHub MCP Server...")
    

    try:
         async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
            print("\n正在验证 Token 身份信息...")
            user_info = await session.call_tool(
            "search_users",
             arguments={
                "query": "fanfande"  # 填入你自己的 GitHub 用户名
            }
            )
            print("身份验证成功！")

    except Exception as e:
        print(f"身份验证失败，当前的 Token 可能无效：{e}")


if __name__ == "__main__":
    # Windows 下异步策略有时需特殊处理
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(run_github_agent())