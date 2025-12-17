# 需要导入新的 Memory 和 Message 类
from core.memory import Memory
from core.message import UserMessage, SystemMessage, ToolMessage
from engine.tool import function_to_schema ,ToolRegistry
from openai.types.chat import ChatCompletionMessageFunctionToolCall
from openai.types.chat import ChatCompletionMessageToolCall
# ... 其他原有导入

'''
agent 类的组成
使用的LLM模型：
全部的工具函数：
memory模块：



'''
class Agent:
    def __init__(self, llm, system_prompt: str = "你是一个有用的 AI 助手。"):
        self.llm = llm
        #初始化工具模块
        self.toolregistry = ToolRegistry()
        # 初始化记忆模块
        self.memory = Memory(
            system_message=SystemMessage(system_prompt),
            max_messages=15 # 设定一个合理的窗口值
        )

        #

    # # tool 装饰器保持不变...
    # def tool(self,func:Callable):
    #     """
    #     装饰器核心逻辑
    #     使用方式: 
    #     @agent.tool
    #     def my_func(...): ...
    #     """
    #     # 1. 【翻译】解析函数的 Schema (给 LLM 看的菜单)
    #     schema = function_to_schema(func)
    #     self.tool_schemas.append(schema)
        
    #     # 2. 【入库】保存函数的可执行对象 (给 Agent 执行用的工具箱)
    #     # 使用函数名作为 Key
    #     self.tool_map[func.__name__] = func
        
    #     # 3. 【归还】必须返回原函数，否则 Python 代码里就没法正常调用这个函数了
    #     return func
    

    # _execute_tool 方法保持不变...
    def _execute_tool(self,tool_call:ChatCompletionMessageFunctionToolCall):
        fun = self.toolregistry.get_tool(tool_call.function.name)



    def run(self, prompt: str, max_turns: int = 5):
        # 1. 将用户输入存入记忆
        self.memory.add(UserMessage(prompt))
        
        current_turn = 0
        while current_turn < max_turns:
            current_turn += 1
            
            # 2. 从 Memory 获取整理好的消息列表（已自动处理滑动窗口）
            messages_to_send = self.memory.to_messages()
            print(messages_to_send)
            print(self.toolregistry.tool_schemas)
            # 3. 调用 LLM
            # 注意：你的 LLM 类现在返回的是 LLMMessage 或 ChatCompletionMessage
            response_message = self.llm.chat(messages_to_send, tools=self.toolregistry.tool_schemas)
            
            print(response_message)
            # 4. 将 AI 的回复存入记忆
            # 如果 response_message 是 OpenAI 原生对象，需要适配一下存入 Memory
            # 假设你的 LLM.chat 已经按照刚才的建议，返回了包含 tool_calls 的对象
            
            # 这里的 response_message 可能是 OpenAI 的原生对象，我们需要把它存进 Memory
            # 为了简单，直接存入（因为我们 Memory 存的是 Message 对象，如果是原生对象可能需要转换，
            # 但为了兼容你之前的 LLMMessage，这里假设 LLM 返回的是 LLMMessage）
            
            # 【重要】将 LLM 的回复加入 Memory
            self.memory.add(response_message)

            # 5. 判断是否调用工具
            if response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    # 执行工具
                    tool_result_dict = self._execute_tool(tool_call)
                    
                    # 封装为 ToolMessage 对象
                    tool_msg = ToolMessage(
                        content=tool_result_dict["content"],
                        tool_call_id=tool_result_dict["tool_call_id"]
                    )
                    
                    # 将工具结果存入记忆
                    self.memory.add(tool_msg)
                
                # 循环继续，下一轮 LLM 会看到工具结果
            else:
                print(f"🤖 Answer: {response_message.content}")
                return response_message.content
        
        return "Max turns reached."