from mcp.server.fastmcp import FastMCP
import subprocess
import platform

# Create an MCP server
bashmcp = FastMCP("bash")

@bashmcp.tool()
def execute_shell(command: str, shell_type: str = "cmd") -> str:
    """
    执行 Shell 命令并返回结果
    :param command: 要执行的命令字符串
    :param shell_type: 'cmd' 或 'powershell' (Windows下)
    """
    try:
        # 构建命令参数
        if platform.system() == "Windows":
            if shell_type.lower() == "powershell":
                # PowerShell 需要特殊的调用方式
                args = ["powershell", "-Command", command]
            else:
                # CMD
                args = ["cmd", "/c", command]
        else:
            # Linux/Mac 默认用 bash/sh
            args = ["/bin/sh", "-c", command]

        # 执行命令
        # capture_output=True 捕获标准输出和错误输出
        # text=True 将输出解码为字符串
        result = subprocess.run(
            args, 
            capture_output=True, 
            text=True, 
            timeout=30  # 设置超时防止死循环
        )

        # 组合输出
        output = result.stdout
        if result.stderr:
            output += f"\n[Errors]:\n{result.stderr}"
            
        return output.strip()

    except subprocess.TimeoutExpired:
        return "Error: Command timed out."
    except Exception as e:
        return f"Error executing command: {str(e)}"
    
# --- 修正点 3: 必须加上这一段！否则服务不会启动 ---
if __name__ == "__main__":
    #print("Starting Bash MCP Server...", flush=True) # 打印个日志证明跑起来了
    bashmcp.run()