# agent_core.py
import os
from typing import Optional, Any, Dict, List

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from tools_search import build_tools
from memory_store import build_memory, WindowMemory


def build_agent(
    knowledge_path: Optional[str] = None,
    memory_k: int = 3,
    model: str = "gpt-4o",
    temperature: float = 0.2,
    api_key: Optional[str] = "",
):
    key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise EnvironmentError("Missing OPENAI_API_KEY (or pass api_key=...)")

    llm = ChatOpenAI(model=model, temperature=temperature, api_key=key)

    tools = build_tools(path=knowledge_path) if knowledge_path else build_tools()

    # LangChain v1: create_agent -> 返回一个可 invoke 的 agent（runnable）
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "你是一个有用的助手。"
            "当用户的问题需要查询本地资料/定义/背景,或者你不知道问题答案时，调用 local_search 工具。"
            "如果不需要查资料就直接回答。"
        ),
    )

    memory: WindowMemory = build_memory(k=memory_k)
    return agent, memory


def run_one_turn(agent, memory: WindowMemory, user_text: str) -> str:
    # 取最近 k 轮历史 + 当前输入，交给 agent
    history: List[Dict[str, str]] = memory.get()
    messages = history + [{"role": "user", "content": user_text}]

    result: Dict[str, Any] = agent.invoke({"messages": messages})
    # v1 通常会返回 {"messages": [...]}，最后一条是 assistant
    out_messages = result.get("messages", [])
    if not out_messages:
        return str(result)

    last = out_messages[-1]
    # last 可能是 dict，也可能是 message object；都兼容一下
    text = last.get("content") if isinstance(last, dict) else getattr(last, "content", str(last))
    return text
