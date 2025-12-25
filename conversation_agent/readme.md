## 1. 项目简介

本项目实现了一个**基础通用对话 Agent**，支持多轮对话与工具调用能力。  
Agent 可以根据用户问题**自动决定是否调用搜索工具**，并基于本地文档检索结果生成回答。

项目基于 **Python + LangChain** 实现。

---

## 2. 功能概览

### 2.1 多轮对话能力
- 支持连续多轮对话
- 保留最近 2–3 轮上下文信息（窗口记忆）
- 避免上下文无限增长

### 2.2 搜索能力（Local Search Tool）
- 提供 `local_search` 搜索工具
- 对本地 `knowledge.json` 文档进行关键词匹配
- 返回 Top-K 相关内容作为参考信息

### 2.3 自动工具调用（Agentic Behavior）
- Agent 使用 LLM 的 tool-calling 能力
- 根据用户问题自主决定是否调用搜索工具
- 搜索结果会被纳入最终回答生成过程

---

## 3. 项目结构说明

├── app.py # 项目入口：运行对话循环
├── agent_core.py # Agent 核心逻辑（LLM + Tool + Memory）
├── tools_search.py # 本地搜索工具（local_search）
├── memory_store.py # 对话记忆模块（窗口记忆）
├── knowledge.json # 本地知识库（JSON 格式）
└── README.md


### 3.1 各文件功能说明

- **app.py**  
  项目运行入口，负责：
  - 接收用户输入
  - 调用 Agent 执行单轮对话
  - 输出模型回复  
  同时维护对话循环，支持多轮交互。

- **agent_core.py**  
  Agent 的核心模块，主要功能包括：
  - 初始化大语言模型（GPT API）
  - 注册可用工具（如本地搜索工具）
  - 组装 Agent（基于 tool-calling 机制）
  - 提供单轮对话调用接口  
  是整个系统的“中枢”。

- **tools_search.py**  
  搜索工具模块，定义 `local_search` 工具：
  - 从 `knowledge.json` 加载本地文档
  - 使用关键词匹配进行简单检索
  - 返回 Top-K 相关片段供 Agent 使用  
  该工具通过 LangChain 的 `@tool` 装饰器注册，供 Agent 自主调用。

- **memory_store.py**  
  对话记忆模块，采用窗口记忆（Window Memory）策略：
  - 仅保留最近 *k* 轮对话（user + assistant）
  - 防止上下文无限增长
  - 保证多轮对话的连贯性

- **knowledge.json**  
  本地知识库文件：
  - 使用 JSON 格式存储文档
  - 每条文档包含 `id / title / tags / text`
  - 可自由扩展，用于模拟检索场景