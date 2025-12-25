
from agent_core import build_agent, run_one_turn


def main():
    agent, memory = build_agent(
        memory_k=3,
        model="gpt-4o",
        temperature=0.2,
        
        api_key=""
    )

    print("✅ Basic Agent 已启动。输入 quit/exit 退出。\n")

    while True:
        user_text = input("You: ").strip()
        if not user_text:
            continue
        if user_text.lower() in ("quit", "exit"):
            print("Bye!")
            break

        try:
            answer = run_one_turn(agent, memory, user_text)
            print(f"Assistant: {answer}\n")

            # 更新窗口记忆（保留最近 k 轮）
            memory.add_user(user_text)
            memory.add_ai(answer)

        except Exception as e:
            print(f"[ERROR] {e}\n")


if __name__ == "__main__":
    main()
