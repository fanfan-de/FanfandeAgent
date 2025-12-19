import os
import requests
import math
import json
from pydantic import BaseModel, Field
from typing import Callable, Type, Any, Dict

# --- 1. 定义工具基类和装饰器 (与之前设计保持一致) ---
class BaseTool:
    name: str
    description: str
    args_schema: Type[BaseModel]
    func: Callable
    is_sensitive: bool = False # 增加一个敏感操作标记

    def __init__(self, name: str, description: str, func: Callable, args_schema: Type[BaseModel], is_sensitive: bool = False):
        self.name = name
        self.description = description
        self.func = func
        self.args_schema = args_schema
        self.is_sensitive = is_sensitive

    def to_schema(self) -> dict:
        """转为 OpenAI Function Calling 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_schema.model_json_schema()
            }
        }
    
    def run(self, **kwargs) -> Any:
        """执行工具，并进行参数校验"""
        try:
            # Pydantic 自动进行参数校验和类型转换
            validated_args = self.args_schema(**kwargs)
            return self.func(**validated_args.model_dump())
        except Exception as e:
            raise ValueError(f"Tool {self.name} argument validation or execution error: {e}")

# 装饰器简化工具定义
def create_tool(name: str, description: str, args_schema: Type[BaseModel], is_sensitive: bool = False):
    def decorator(func: Callable):
        return BaseTool(name=name, description=description, func=func, args_schema=args_schema, is_sensitive=is_sensitive)
    return decorator

# --- 2. 定义具体的工具和其 Pydantic 参数 Schema ---

# --- 工具 1: 文件读取 ---
class ReadFileArgs(BaseModel):
    file_path: str = Field(..., description="要读取的文件的路径。")

@create_tool(
    name="read_file",
    description="从指定路径读取文件内容。",
    args_schema=ReadFileArgs
)
def _read_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        return f"错误: 文件 '{file_path}' 不存在。"
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        return f"错误: 读取文件 '{file_path}' 失败: {str(e)}"

# --- 工具 2: 文件写入 (敏感操作) ---
class WriteFileArgs(BaseModel):
    file_path: str = Field(..., description="要写入的文件路径。")
    content: str = Field(..., description="要写入文件的内容。")
    append: bool = Field(False, description="如果为True，则追加内容而不是覆盖。")

@create_tool(
    name="write_file",
    description="向指定路径写入内容。如果文件不存在则创建。这是一个敏感操作，可能需要人工确认。",
    args_schema=WriteFileArgs,
    is_sensitive=True # 标记为敏感
)
def _write_file(file_path: str, content: str, append: bool) -> str:
    mode = 'a' if append else 'w'
    try:
        with open(file_path, mode, encoding='utf-8') as f:
            f.write(content)
        return f"成功写入文件 '{file_path}'。"
    except Exception as e:
        return f"错误: 写入文件 '{file_path}' 失败: {str(e)}"

# --- 工具 3: 计算器 ---
class CalculatorArgs(BaseModel):
    expression: str = Field(..., description="要计算的数学表达式字符串，例如 '10 * (2 + 3)'。")

@create_tool(
    name="calculator",
    description="评估一个简单的数学表达式，支持加减乘除和括号。",
    args_schema=CalculatorArgs
)
def _calculator(expression: str) -> str:
    try:
        # 使用 eval() 需要非常小心，在生产环境中应使用更安全的解析器
        # 这里仅用于测试目的
        result = eval(expression, {"__builtins__": None}, {"math": math}) 
        return str(result)
    except Exception as e:
        return f"错误: 计算表达式失败: {str(e)}"

# --- 工具 4: 发送 HTTP GET 请求 ---
class HttpGetArgs(BaseModel):
    url: str = Field(..., description="要请求的URL。")
    params: Dict[str, str] = Field({}, description="URL参数字典。")

@create_tool(
    name="http_get",
    description="向指定的URL发送HTTP GET请求并返回响应文本。可用于获取网页内容或API数据。",
    args_schema=HttpGetArgs
)
def _http_get(url: str, params: Dict[str, str]) -> str:
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status() # 检查HTTP错误
        return response.text
    except requests.exceptions.Timeout:
        return f"错误: 请求超时: {url}"
    except requests.exceptions.RequestException as e:
        return f"错误: HTTP GET 请求失败: {str(e)}"

# --- 工具 5: 获取当前日期 ---
class GetCurrentDateArgs(BaseModel):
    format: str = Field("%Y-%m-%d", description="日期格式，例如 '%Y-%m-%d'。")

@create_tool(
    name="get_current_date",
    description="获取当前日期，可以指定输出格式。",
    args_schema=GetCurrentDateArgs
)
def _get_current_date(format: str) -> str:
    from datetime import datetime
    return datetime.now().strftime(format)

# --- 3. 收集所有工具 ---
ALL_TOOLS = [
    _read_file,
    _write_file,
    _calculator,
    _http_get,
    _get_current_date
]

# --- 4. 模拟工具注册中心 ---
class MockToolRegistry:
    def __init__(self, tools: List[BaseTool]):
        self._tools: Dict[str, BaseTool] = {tool.name: tool for tool in tools}

    def get_tool_schemas(self) -> List[dict]:
        """获取所有工具的 OpenAPI 兼容 Schema"""
        return [tool.to_schema() for tool in self._tools.values()]

    def run_tool(self, tool_name: str, args: Dict[str, Any], human_approval: bool = False) -> Any:
        tool = self._tools.get(tool_name)
        if not tool:
            return f"错误: 未知的工具 '{tool_name}'。"
        
        if tool.is_sensitive and not human_approval:
            raise PermissionError(f"工具 '{tool_name}' 是敏感操作，需要人工确认。")
        
        return tool.run(**args)

# 实例化注册中心
mock_tool_registry = MockToolRegistry(ALL_TOOLS)

# --- 测试用例 (你可以在你的 AgentExecutor 中调用这些) ---
async def test_tool_calling():
    print("--- 测试工具调用 ---")

    # 1. 测试文件读取
    print("\n测试 read_file:")
    try:
        content = mock_tool_registry.run_tool("read_file", {"file_path": "example.txt"})
        print(f"文件内容: {content.strip()}")
    except Exception as e:
        print(f"错误: {e}")

    # 2. 测试计算器
    print("\n测试 calculator:")
    try:
        result = mock_tool_registry.run_tool("calculator", {"expression": "15 * (8 - 3)"})
        print(f"计算结果: {result}")
    except Exception as e:
        print(f"错误: {e}")

    # 3. 测试 HTTP GET
    print("\n测试 http_get:")
    try:
        # 尝试请求一个真实且简单的API
        # 由于我们无法直接在测试中解析HTTP响应的复杂性
        # 这里只测试是否能成功获取到文本
        # 注意：不要请求太大的网站或需要认证的网站
        response_text = mock_tool_registry.run_tool("http_get", {"url": "https://httpbin.org/get", "params": {"test": "value"}})
        print(f"HTTP GET 响应前100字符: {response_text[:100]}...")
    except Exception as e:
        print(f"错误: {e}")

    # 4. 测试获取当前日期
    print("\n测试 get_current_date:")
    try:
        today = mock_tool_registry.run_tool("get_current_date", {"format": "%Y年%m月%d日"})
        print(f"今天的日期: {today}")
    except Exception as e:
        print(f"错误: {e}")

    # 5. 测试文件写入 (敏感操作，需要人工确认)
    print("\n测试 write_file (敏感操作):")
    try:
        mock_tool_registry.run_tool("write_file", {"file_path": "test_output.txt", "content": "这是测试写入的内容。"}, human_approval=True)
        print("成功模拟写入文件 (需要 human_approval=True)。")
        # 验证写入
        content_written = mock_tool_registry.run_tool("read_file", {"file_path": "test_output.txt"})
        print(f"写入后读取: {content_written.strip()}")
    except PermissionError as e:
        print(f"预期错误: {e}")
    except Exception as e:
        print(f"错误: {e}")

    # 尝试不带人工确认的写入
    print("\n测试 write_file (未授权写入):")
    try:
        mock_tool_registry.run_tool("write_file", {"file_path": "test_output_no_auth.txt", "content": "这应该失败。"})
    except PermissionError as e:
        print(f"预期错误 (未授权): {e}")
    except Exception as e:
        print(f"意外错误: {e}")


# 运行测试
if __name__ == "__main__":
    asyncio.run(test_tool_calling()) # 如果你的 AgentExecutor 是异步的，也需要用 asyncio.run