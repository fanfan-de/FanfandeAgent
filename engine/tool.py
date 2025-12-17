import inspect
from typing import get_type_hints,Dict,Callable,List

def get_json_type(py_type):
    """
    将 Python 类型映射为 JSON Schema 类型
    """
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    # 如果是更复杂的类型（如 Optional），这里可以扩展逻辑
    # 简单起见，如果找不到映射，默认当做 string 处理
    return type_map.get(py_type, "string")

#输入函数，输出他的schema
def function_to_schema(func) -> dict:
    """
    核心魔法：将 Python 函数转换为 OpenAI Function JSON Schema
    """
    # 1. 获取函数名和文档
    name = func.__name__
    description = (func.__doc__ or "").strip()

    # 2. 解析参数
    sig = inspect.signature(func)#这是 Python 的内省神器，它能告诉你函数有哪些参数、默认值是什么。
    type_hints = get_type_hints(func) # 获取类型提示
    
    parameters = {
        "type": "object",
        "properties": {},
        "required": []
    }

    for param_name, param in sig.parameters.items():
        # 跳过 self 和 cls (如果是类方法)
        if param_name == 'self' or param_name == 'cls':
            continue

        # 获取参数类型
        annotation = type_hints.get(param_name, str) # 默认为 str
        json_type = get_json_type(annotation)

        # 构建参数描述
        parameters["properties"][param_name] = {
            "type": json_type,
            "description": f"Parameter {param_name}" # 可以在这里做更多优化，解析 docstring 获取参数说明
        }

        # 判断是否为必填项 (没有默认值就是必填)
        if param.default == inspect.Parameter.empty:
            parameters["required"].append(param_name)

    # 3. 组装最终结构
    schema = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        }
    }
    return schema


class ToolRegistry:

    
    """工具注册器"""
    def __init__(self):
        self._tools: Dict[str, Dict] = {}
        self.tool_schemas:List[Dict[str, Dict]] = []
        self.tool_map:Dict[str,Callable]={}
    
    
    def register(self,func):
        # 1. 【翻译】解析函数的 Schema (给 LLM 看的菜单)
        schema = function_to_schema(func)
        self.tool_schemas.append(schema)

        # 2. 【入库】保存函数的可执行对象 (给 Agent 执行用的工具箱)
        # 使用函数名作为 Key
        self.tool_map[func.__name__] = func
        return func

    def get_tool(self, name: str):
        """获取工具"""
        return self.tool_map.get(name)
    
    def list_tools(self):
        """列出所有工具"""
        return self._tools





