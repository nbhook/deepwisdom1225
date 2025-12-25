import json
import os
from typing import List, Dict, Any, Tuple

from langchain.tools import tool

DEFAULT_KNOWLEDGE_PATH = os.path.join(os.path.dirname(__file__), "knowledge.json")


def load_knowledge(path: str = DEFAULT_KNOWLEDGE_PATH) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"knowledge file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("knowledge.json must be a list of chunks")

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"chunk #{i} is not an object")
        for key in ("id", "title", "tags", "text"):
            if key not in item:
                raise ValueError(f"chunk #{i} missing key: {key}")
    return data


def _score_chunk(query: str, chunk: Dict[str, Any]) -> int:
    q = (query or "").lower().strip()
    if not q:
        return 0

    hay = " ".join([
        str(chunk.get("id", "")),
        str(chunk.get("title", "")),
        " ".join(chunk.get("tags", []) or []),
        str(chunk.get("text", "")),
    ]).lower()

    tokens = [t for t in q.split() if t]
    if not tokens:
        tokens = [q]

    score = 0
    for t in tokens:
        if t in hay:
            score += 1
    return score


def local_search_impl(
    query: str,
    top_k: int = 3,
    max_chars: int = 220,
    path: str = DEFAULT_KNOWLEDGE_PATH
) -> str:
    knowledge = load_knowledge(path=path)

    scored: List[Tuple[int, Dict[str, Any]]] = []
    for chunk in knowledge:
        s = _score_chunk(query, chunk)
        if s > 0:
            scored.append((s, chunk))

    if not scored:
        return "未在本地知识库中找到相关内容。"

    scored.sort(key=lambda x: x[0], reverse=True)
    hits = scored[:top_k]

    lines: List[str] = []
    for s, chunk in hits:
        text = str(chunk.get("text", "")).strip().replace("\n", " ")
        snippet = text[:max_chars] + ("..." if len(text) > max_chars else "")
        lines.append(f"[{chunk.get('id')}] {chunk.get('title')}\n{snippet}")

    return "\n\n".join(lines)


def build_tools(path: str = DEFAULT_KNOWLEDGE_PATH):
    """
    LangChain v1+ style: tools are functions decorated with @tool.
    We create a closure to bind the knowledge path.
    """

    @tool("local_search")
    def local_search(query: str) -> str:
        """从本地 knowledge.json 检索信息。输入应为关键词或简短问题。需要查资料/定义/背景时使用。"""
        return local_search_impl(query=query, top_k=3, max_chars=220, path=path)

    return [local_search]
