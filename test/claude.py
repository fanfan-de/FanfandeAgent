import anthropic
import os

def test_claude_chat():
    # 1. 设置 API Key
    # 建议将 API Key 存放在环境变量中，或者直接在此处替换 'your-api-key-here'
    api_key = os.getenv("ANTHROPIC_API_KEY") or "在此输入你的API_KEY"

    if not api_key or api_key == "在此输入你的API_KEY":
        print("错误: 请设置有效的 ANTHROPIC_API_KEY")
        return

    # 2. 初始化客户端
    client = anthropic.Anthropic(api_key=api_key)

    skills = client.beta.skills.list(
        source="anthropic",
        betas=["skills-2025-10-02"]
    )

    try:
        print("正在发送请求给 Claude...\n")
        
        # 3. 创建消息
        # 常用模型: 
        # - claude-3-5-sonnet-20240620 (目前最强最推荐)
        # - claude-3-opus-20240229
        # - claude-3-haiku-20240307
        message = client.beta.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": "你好，Claude！请简短地介绍一下你自己，并确认你现在的运行状态。"}
            ]
            container={
                "skills": [
                    {
                        "type": "anthropic",
                        "skill_id": "pptx",
                        "version": "latest"
                    }
                ]
            },
        )

        # 4. 打印响应结果
        print("--- Claude 的回复 ---")
        print(message.content[0].text)
        print("\n--- 测试完成 ---")

    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    test_claude_chat()


