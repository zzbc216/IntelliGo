"""
IntelliGo CLI 入口
基于 LangGraph 的多模态主动规划助手
"""
import sys
import os
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from config import config
from graph.builder import get_compiled_graph
from graph.state import GraphState

from memory.vector_store import UserMemory
console = Console()


def print_banner():
    """打印欢迎横幅"""
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║   🚀 IntelliGo - 智能出行助手                              ║
    ║                                                          ║
    ║   基于 LangGraph 的多模态主动规划系统                         ║
    ║                                                          ║
    ║   功能:                                                   ║
    ║   • 📍 多城市/多日行程规划                                   ║
    ║   • 👔 基于天气的穿搭建议                                    ║
    ║   • 🧠 用户偏好学习与记忆                                    ║
    ║   • ⚠️ 恶劣天气主动预警与备选方案                             ║
    ║                                                          ║
    ║   输入 'quit' 或 'exit' 退出                               ║
    ╚══════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="cyan")

def validate_config():
    """验证配置"""
    errors = config.validate()
    if errors:
        for err in errors:
            console.print(err)
        if "OPENAI_API_KEY" in str(errors):
            console.print("\n[red]请在 .env 文件中设置 OPENAI_API_KEY[/red]")
            sys.exit(1)
    console.print("✅ 配置验证通过\n", style="green")


def run_cli():
    """运行 CLI 交互循环"""
    print_banner()
    validate_config()

    # 编译图
    console.print("🔧 正在初始化 LangGraph...", style="yellow")
    graph = get_compiled_graph()
    console.print("✅ 系统就绪！\n", style="green")

    # 示例提示
    console.print("[dim]💡 试试这些问法:[/dim]")
    console.print("[dim]   • 周末想去杭州玩两天，喜欢安静的地方[/dim]")
    console.print("[dim]   • 明天北京穿什么合适？[/dim]")
    console.print("[dim]   • 帮我规划上海到苏州的三日游，预算2000[/dim]")
    console.print()

    # ✅ 跨轮对话状态（解决“忘记上次城市/偏好”等问题）
    last_state = None

    while True:
        try:
            # 获取用户输入
            user_input = Prompt.ask("\n[bold blue]你[/bold blue]")

            if user_input.lower() in ["quit", "exit", "q", "退出"]:
                console.print("\n👋 再见！祝您旅途愉快！", style="cyan")
                break

            if not user_input.strip():
                continue

            # ====== 可视化命令（不走图）======
            cmd = user_input.strip().lower()

            if cmd in ["/state", "state"]:
                if not last_state:
                    console.print("[yellow]当前还没有会话状态（先问一句再试）[/yellow]")
                    continue

                intent = last_state.get("intent")
                entities = last_state.get("entities") or {}

                md = "\n".join([
                    "## 当前会话状态（state）",
                    f"- 意图: {getattr(intent, 'intent_type', None)}",
                    f"- 城市: {entities.get('cities')}",
                    f"- 天数: {entities.get('duration_days')}",
                    f"- 日期: {entities.get('dates')}",
                    f"- 偏好: {entities.get('preferences')}",
                ])
                console.print(Panel(Markdown(md), title="[bold green]Debug[/bold green]", border_style="green"))
                continue

            if cmd in ["/profile", "profile"]:
                if not last_state:
                    console.print("[yellow]当前还没有用户画像（先问一句再试）[/yellow]")
                    continue

                up = last_state.get("user_profile") or {}
                profile = up.get("profile")
                rel = up.get("relevant_memories") or []

                lines = ["## 用户画像（user_profile）", "### Profile"]
                lines.append(f"```json\n{profile}\n```" if profile is not None else "- (empty)")

                lines.append("\n### Relevant memories (top-k)")
                if not rel:
                    lines.append("- (empty)")
                else:
                    for m in rel:
                        lines.append(
                            f"- **{m.get('category','unknown')}** / {m.get('type','unknown')} / score={m.get('score')}: {m.get('content')}"
                        )

                console.print(Panel(Markdown("\n".join(lines)), title="[bold green]Profile[/bold green]", border_style="green"))
                continue

            if cmd.startswith("/clear"):
                parts = user_input.strip().split(maxsplit=1)
                token = parts[1].strip() if len(parts) == 2 else ""

                if not getattr(config, "purge_token", ""):
                    console.print("[red]PURGE_TOKEN 未设置，禁止清空操作。[/red]")
                    continue

                if token != config.purge_token:
                    console.print("[red]口令错误：未执行清空。[/red]")
                    continue

                UserMemory.clear_all_persisted_data()

                # 重置 graph/nodes.py 中的全局 user_memory 实例
                from graph import nodes
                nodes.user_memory.reset()

                last_state = None  # 同时重置 CLI 的对话状态
                console.print("[green]✅ 已清空全部用户画像/记忆（Chroma），并重置当前会话。[/green]")

                console.print(f"[dim]cwd={os.getcwd()}[/dim]")
                console.print(f"[dim]persist_dir={config.chroma_persist_dir}[/dim]")

                continue
            # ====== 可视化命令结束 ======


            # ✅ 构建初始状态：继承上一轮的部分字段 + 重置本轮字段
            carry = {}
            if last_state:
                for k in ["messages", "entities", "user_profile", "weather_data", "trip_plan", "excluded_places", "included_places"]:
                    v = last_state.get(k)
                    if v is not None:
                        carry[k] = v

            reset = dict(
                rewritten_query="",
                rewrite_slots={},
                need_clarification=False,
                clarifying_questions=[],
                clarify_only=False,
                intent=None,
                clothing_advice="",
                final_response="",
                current_node="",
                needs_replan=False,
                error_message="",
            )

            initial_state = GraphState(**carry, **reset, user_input=user_input)

            # 运行图
            console.print("\n[dim]🔄 思考中...[/dim]")

            final_state = None
            for event in graph.stream(initial_state, stream_mode="values"):
                final_state = event
                if config.debug and event.get("current_node"):
                    console.print(f"[dim]  → {event['current_node']}[/dim]")

            # 输出结果
            if final_state and final_state.get("final_response"):
                response = final_state["final_response"]
                console.print()
                console.print(
                    Panel(
                        Markdown(response),
                        title="[bold green]IntelliGo[/bold green]",
                        border_style="green",
                    )
                )

                # ✅ 调试信息：打印“最终 state.entities”，不要只看 intent.extracted_entities
                if config.debug:
                    if final_state.get("intent"):
                        intent = final_state["intent"]
                        console.print(
                            f"\n[dim]🎯 意图: {intent.intent_type} (置信度: {intent.confidence})[/dim]"
                        )

                    if final_state.get("entities") is not None:
                        console.print(f"[dim]📦 实体(state.entities): {final_state['entities']}[/dim]")

            else:
                console.print("[red]抱歉，处理过程中出现问题[/red]")

            # ✅ 保存本轮状态，供下一轮继承
            last_state = final_state

        except KeyboardInterrupt:
            console.print("\n\n👋 再见！", style="cyan")
            break
        except Exception as e:
            console.print(f"\n[red]❌ 错误: {e}[/red]")
            if config.debug:
                import traceback

                console.print(traceback.format_exc())


if __name__ == "__main__":
    run_cli()

