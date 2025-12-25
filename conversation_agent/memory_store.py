# memory_store.py
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class WindowMemory:
    k: int = 3
    # 存 messages：每条是 {"role": "...", "content": "..."}
    messages: List[Dict[str, str]] = field(default_factory=list)

    def add_user(self, text: str):
        self.messages.append({"role": "user", "content": text})
        self._trim()

    def add_ai(self, text: str):
        self.messages.append({"role": "assistant", "content": text})
        self._trim()

    def get(self) -> List[Dict[str, str]]:
        return list(self.messages)

    def _trim(self):
        # 1轮 = user+assistant 两条消息；保留最近 k 轮 => 2k 条
        max_len = 2 * self.k
        if len(self.messages) > max_len:
            self.messages = self.messages[-max_len:]


def build_memory(k: int = 3) -> WindowMemory:
    return WindowMemory(k=k)
