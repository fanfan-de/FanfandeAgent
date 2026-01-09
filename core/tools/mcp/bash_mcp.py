from mcp.server.fastmcp import FastMCP
import subprocess
import platform
import sys
import os

# Create an MCP server
bashmcp = FastMCP("bash")

# 定义一个全局的日志文件路径
LOG_FILE = "agent_cmd_output.log"

def _open_monitor_window():
    """
    辅助函数：弹出一个新窗口，实时监控日志文件
    """
    # 确保文件存在
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write("Waiting for command...\n")

    system_name = platform.system()
    
    try:
        if system_name == "Windows":
            # Windows: 使用 start cmd /k powershell Get-Content ... -Wait
            # /k 保持窗口打开
            # Get-Content -Wait 相当于 tail -f
            monitor_cmd = f'start "Agent Console Monitor" cmd /k "powershell Get-Content {LOG_FILE} -Wait"'
            subprocess.Popen(monitor_cmd, shell=True)
            
        elif system_name == "Darwin": # macOS
            # macOS: 使用 open -a Terminal
            subprocess.Popen(["open", "-a", "Terminal", f"tail -f {LOG_FILE}"])
            
        elif system_name == "Linux":
            # Linux (尝试常见的终端模拟器)
            # 你可能需要根据实际环境调整，如 gnome-terminal, xterm 等
            subprocess.Popen(["x-terminal-emulator", "-e", f"tail -f {LOG_FILE}"])
            
    except Exception as e:
        print(f"[Warning] Could not open monitor window: {e}")

@bashmcp.tool()
def execute_shell(command: str, shell_type: str = "cmd") -> str:
    """
    执行 Shell 命令，在独立窗口显示输出，并返回结果
    """
    # 1. 准备工作：清空或标记日志文件
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"--- Executing: {command} ---\n")
    
    # 可选：如果希望每次都自动弹窗，取消下面这行的注释。
    # 如果你不想满屏弹窗，可以手动开一个窗口运行 `Get-Content agent_cmd_output.log -Wait`，然后注释掉这行。
    _open_monitor_window()

    try:
        # 2. 构建命令参数
        if platform.system() == "Windows":
            if shell_type.lower() == "powershell":
                args = ["powershell", "-Command", command]
            else:
                args = ["cmd", "/c", command]
        else:
            args = ["/bin/sh", "-c", command]

        # 3. 启动进程
        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # 将错误也重定向到标准输出
            text=True,
            bufsize=1, 
            encoding='utf-8',
            errors='replace'
        )

        full_output = []
        
        # 4. 实时读取并写入文件
        # 使用 append 模式 ('a') 打开文件
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            
            if line:
                # A. 写入日志文件（让那个新窗口显示）
                with open(LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(line)
                    f.flush() # 强制刷新缓冲区，确保那边立即看到
                
                # B. 存入内存（给 AI 用）
                full_output.append(line)

        # 5. 处理结果
        return_code = process.poll()
        final_result = "".join(full_output)
        
        # 写入结束标记
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n[Finished with code {return_code}]\n")

        if return_code != 0:
            final_result += f"\n[Process exited with code {return_code}]"

        return final_result.strip()

    except Exception as e:
        err_msg = f"Error executing command: {str(e)}"
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n{err_msg}\n")
        return err_msg
    
# --- 修正点 3: 必须加上这一段！否则服务不会启动 ---
if __name__ == "__main__":
    #print("Starting Bash MCP Server...", flush=True) # 打印个日志证明跑起来了
    bashmcp.run()