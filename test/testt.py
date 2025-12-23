import asyncio
import os
import traceback  # 引入堆栈打印工具
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def debug_github_agent():
    # 1. 确保 Token 字符串前后没有空格
    github_token = "" # 换成你的真实 Token

    server_params = StdioServerParameters(
        command="npx.cmd",
        args=["--registry", "https://registry.npmmirror.com", "-y", "@modelcontextprotocol/server-github"],
        env={
            # 2. 检查环境变量：确保没有引入 os.environ 以排除系统旧 Token 的干扰
            "GITHUB_PERSONAL_ACCESS_TOKEN": github_token,
            "PATH": os.environ.get("PATH", "") # 必须保留 PATH 否则找不到 npx
        }
    )

    print("🚀 开始调试模式...")
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print("✅ 管道连接成功")

                response = await session.list_tools()

                available_tools = [t.name for t in response.tools]
                print(f"当前 Server 真正支持的工具列表: {available_tools}")

                # 3. 关键：尝试调用一个绝对存在的工具
                # 并故意不加 try-except 看看它到底报什么错
                print("\n尝试搜索仓库...")
                
                # 注意：arguments 必须是一个 dict
                result = await session.call_tool(
                    "search_repositories", 
                    arguments={"query": "mcp", "page": 1}
                )
                
                print("🎉 成功获取数据！")
                print(f"结果: {str(result.content)[:100]}...")

                user_info = await session.call_tool(
                  "search_users",
                arguments={
                    "q": "fanfan-de"  # 填入你自己的 GitHub 用户名
                 }
                )
                print(user_info)
                

                print("身份验证成功！")

    except Exception:
        print("\n❌ 捕获到详细错误堆栈：")
        traceback.print_exc() # 这行会打印出最详细的报错位置和原因

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(debug_github_agent())