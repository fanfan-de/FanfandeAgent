Skills 是 Anthropic 提出的一个编撰Agent能力的文档标准。

格式标准：
https://agentskills.io/specification#directory-structure
翻译后：
### 1. 目录结构 (Directory structure) 
一个技能本质上是一个文件夹，其名称必须与内部定义的技能名一致。 
* **必需文件**：`SKILL.md`（包含技能的核心逻辑和说明）。
* **可选目录**： * `scripts/`：存放可执行的代码脚本（Python, JS 等）。 
* `references/`：存放详细的参考文档（AI 仅在需要时读取）。 
* `assets/`：存放静态资源（模板、图片、数据表）。 
### 2. SKILL.md 格式 
这是技能的“大脑”，由两部分组成：**YAML Frontmatter（元数据）** 和 **Markdown Body（正文指令）**。 
#### **元数据字段说明 (Frontmatter)** 
| 字段              | 必填  | 限制与说明                                          |     |
| :-------------- | :-- | :--------------------------------------------- | --- |
| `name`          | 是   | 1-64字符，仅限小写字母、数字和连字符 `-`。必须与文件夹名一致。            |     |
| `description`   | 是   | 最长1024字符。描述技能做什么以及**何时使用**（这是 AI 判断是否激活技能的关键）。 |     |
| `license`       | 否   | 许可证信息（如 Apache-2.0）。                           |     |
| `compatibility` | 否   | 环境要求（如：需要安装 git, docker 或联网）。                  |     |
| `metadata`      | 否   | 自定义的键值对。                                       |     |
| `allowed-tools` | 否   | （实验性）预先授权该技能可以调用的工具列表。                         |     |

**命名示例：** 
* ✅ 正确：`pdf-processing` 
* ❌ 错误：`PDF-Processing`（不能大写）、`-pdf`（不能以横杠开头）、`pdf--pro`（不能连续横杠）。 

*#### **正文内容 (Body content)** 
在 YAML 之后编写 Markdown。这里没有严格格式限制，主要用于写给 AI 看的“操作指南”。 
* **建议包含**：分步说明、输入输出示例、常见异常情况处理。 

### 3. 可选目录详述 
* **`scripts/`**：AI 可以运行这里的代码。代码应包含良好的错误提示并处理边界情况。 
* **`references/`**：为了节省 Token（上下文），复杂的文档应放在这里。AI 只有在觉得主说明不够用时才会去读这些文件。 
* **`assets/`**：存放配置模板或查找表等原始数据。 
### 4. 渐进式披露原则 (Progressive disclosure) —— **核心设计哲学** 
这是为了在保证 AI 性能的同时节省成本： 
1. **启动时**：仅加载 `name` 和 `description`（约 100 tokens），让 AI 知道有这个技能。
2. **激活后**：当用户需求匹配时，加载整个 `SKILL.md` 的指令。 
3. **按需加载**：只有在执行具体任务时，AI 才会去读取 `scripts/` 或 `references/` 里的具体文件。 
### 5. 文件引用与校验 
* **引用方式**：在 `SKILL.md` 中引用其他文件时，使用相对路径，例如 `scripts/extract.py`。 
* **校验工具**：官方提供了一个工具 `skills-ref`，开发者可以用 `skills-ref validate ./my-skill` 来检查自己的技能包是否符合上述规范。


skills机制 是一种 提示词状态管理的 手段，接收到特定人物的时候，大模型首先得到的是skills文件夹下面所有的 skill.md 的metadata，也就是name 和 description，模型推理出需要某一个能力，再加载对应的完整的skill.md 进入提示词窗口。

> [!被指] Title
> 	skills的做法，是使用 
> 		'LLM 与 prompt窗口 的多次交互，逐次完善当前task所需的prompt'
> 	来替代
> 		'一次性将所有的工具信息，流程信息，背景信息放入prompt窗口'

也就是所谓的”渐进式披露“

| Level                     | When Loaded             | Token Cost            | Content                                                               |
| ------------------------- | ----------------------- | --------------------- | --------------------------------------------------------------------- |
| **Level 1: Metadata**     | Always (at startup)     | ~100 tokens per Skill | `name` and `description` from YAML frontmatter                        |
| **Level 2: Instructions** | When Skill is triggered | Under 5k tokens       | SKILL.md body with instructions and guidance                          |
| **Level 3+: Resources**   | As needed               | Effectively unlimited | Bundled files executed via bash without loading contents into context |
在 claude  skill  的 实践中，
1. skill.md 的 metadata 作为第一层披露，agent执行某一个任务时，所有skill的metadata持久的存在与与LLM对话的每一次 prompt窗口中
```
   什么是md文档的metadata？
```
2. skill.md 文档本身 即为是第二层披露，设计上在执行某个任务期间，不会每次都加入prompt窗口，而是需要由LLM推理出需要加载某个 skill.md 才会由agent将对应的文档 加入到下一次的 prompt窗口之中，（个人猜测，当前task经过多次LLM与agent之间的对话迭代，完成任务之后，prompt窗口会清除掉所有的md文档，只留下metadata？）
3. Resource 包含几乎任何文件形式，一段python代码文本，一段流程prompt文本。dangLLM推理出需要第三层披露之后，会返回给agent 调用信息，由阿根廷去调用resource中的对应内容。这里resource中的内容既可以选择加入prompt窗口，又可以选择直接执行，将结果返回给prompt窗口，完全取决于skill.md 文档中的prompt描述


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

[^1]: 通过LLM与prompt窗口的多次交互，替代一次性将所有的工具信息，流程信息放入prompt窗口，agent在执行一个长期的任务的时候，每一个子任务关注的信息是不一致的，skill的核心是按需获取不同task对应的prompt
