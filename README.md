<!-- PROJECT LOGO -->

# Fanfande_Agent : 从0开始搭建的AI Agent

<br />
<div align="center"
  <p align="center">
`Fanfande_Agent` 是一个追求**完全透明、无黑盒**的 AI Agent 极简实现。该项目拒绝除 LLM 和外部 MCP 服务以外的一切黑盒封装，旨在通过最底层的 Prompt Engineering 和逻辑循环，展现 AI Agent 是如何通过放大 LLM 的推理能力，在现实世界中解决问题的。
  </p>
</div>


### 1. 纯粹的 Context 机制
本项目目前采用最基础、最符合直觉的**消息累加机制**。
- **为什么？** 我们认为消息累加是目前最简单、最透明的上下文管理方式。
- **进化方向：** 针对消息累加带来的 Token 压力，未来将引入**“渐进式披露” (Progressive Disclosure)** 方法作为补充，在保持简单性的同时解决上下文冗余问题。

### 2. 对“多 Agent”的独特见解
**我们认为“多 Agent”是一个伪概念。**
- **本质：** 所谓的多 Agent 协作，本质上是 **Context 分支的状态维护**。
- **拒绝硬编码：** 项目不提供任何“硬编码”的所谓多 Agent 规则方案。
- **方向：** 我们更看好通过维护 Context 状态分支来实现复杂的逻辑流转，而不是通过包装多个 Agent 角色。

### 3. 原生支持 MCP (Model Context Protocol)
项目全面拥抱 MCP，支持通过外部 MCP 服务扩展 Agent 的能力边界，使其能够安全、透明地与外部工具和数据交互。

### 4. 为什么不支持 RAG？
**RAG 是一个“垃圾”（特指将其作为搜索方法提供给 Agent 使用的场景）。**
- 我们拒绝将传统的 RAG 检索链路生搬硬套给 Agent。
- Agent 应该基于逻辑推理和精准的 Context 获取来解决问题，而非在庞杂且质量参差不齐的检索片段中迷失。


<!-- 上手指南 -->
