Skills 是 Anthropic 提出的一个编撰Agent能力的文档标准。

格式标准文档：
https://agentskills.io/specification#directory-structure


skills 属于 提示词状态管理的 一种手段，
当agent执行特定任务task
大模型需要将skills文件夹下 所有的 skill.md 的metadata，填入prompt窗口，
经过模型推理，如果需要某一个skill的详细信息，
再加载对应的完整的skill.md 进入提示词窗口。


> [!备注] 
> 	skills的做法，是使用 
> 		'LLM 与 prompt窗口 的多次交互，逐次完善当前task所需的prompt的过程'
> 	来替代
> 		'一次性将所有的工具信息，流程信息，背景信息放入prompt窗口'

这也就是所谓的”渐进式披露“

skill文件夹 将所有的信息分成了三个层级进行披露

| Level                     | When Loaded             | Token Cost            | Content                                                               |
| ------------------------- | ----------------------- | --------------------- | --------------------------------------------------------------------- |
| **Level 1: Metadata**     | Always (at startup)     | ~100 tokens per Skill | `name` and `description` from YAML frontmatter                        |
| **Level 2: Instructions** | When Skill is triggered | Under 5k tokens       | SKILL.md body with instructions and guidance                          |
| **Level 3+: Resources**   | As needed               | Effectively unlimited | Bundled files executed via bash without loading contents into context |
在 claude  skill  的 实践中，
1. skill.md 的 metadata 作为第一层披露
	   agent执行某一个任务时，
	   所有skill的metadata持久的存在与与LLM对话的每一次 prompt窗口中
```
   什么是md文档的metadata？
```
2. skill.md 文档本身 即为是第二层披露
	   设计上在Agent在执行某个任务期间，skill.md 文档不会每次都加入prompt窗口，
	   而是当LLM推理出需要加载某个 skill.md 时
	   才会由agent将对应的文档 加入到下一次的 prompt窗口之中，
	   （个人猜测，当task完成任务之后，prompt窗口会清除掉所有的md文档，只留下metadata？）
3. Resources文件夹 
	   包含几乎任何文件形式，譬如一段python代码文本，一段流程prompt文本。
	   当LM推理出需要第三层披露之后，会返回给agent 调用信息，由agent去调用Resources中的对应内容。
	   这里Resource中的内容可以
		   1. 加入prompt窗口（文档类）
		   2. 直接执行，将结果返回给prompt窗口（可执行代码类），
		做何选择全取决于skill.md 文档中的prompt描述
	

使用Skills机制的Agent 流程图（个人理解绘制）

![[drawIO-第 2 页.png]]


Skill



Agent Skills 
https://agentskills.io/home
官网
https://claude.com/skills
工程文档 Equipping agents for the real world with Agent Skills
https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills
Claude Docs
https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview
Github NoteBook
https://github.com/anthropics/claude-cookbooks/blob/main/skills/notebooks/01_skills_introduction.ipynb

