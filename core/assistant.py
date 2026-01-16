from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from config import config
from graph.builder import get_compiled_graph
from graph.state import GraphState

# 懒加载/单例：避免每次请求都 compile
_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = get_compiled_graph()
    return _GRAPH


def validate_config_or_raise() -> None:
    """
    只对关键配置抛错：
    - OpenAI API Key（必须）
    - base_url/model 视你的项目是否必须
    AMAP 缺失仅 warning：天气/地图走 Mock。
    """
    errors = config.validate() or []

    fatal = []
    warn = []

    for e in errors:
        if "AMAP_API_KEY" in e:
            warn.append(e)
        else:
            fatal.append(e)

    if warn:
        # 仅提示，不阻断服务
        print("[warn] " + " ; ".join(warn))

    if fatal:
        raise RuntimeError("Config invalid: " + "; ".join(fatal))


def _carry_from_last_state(last_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    复刻你 main.py 的跨轮继承字段逻辑：
    messages/entities/user_profile/weather_data/trip_plan/excluded/included
    """
    carry: Dict[str, Any] = {}
    if not last_state:
        return carry

    for k in [
        "messages",
        "entities",
        "user_profile",
        "weather_data",
        "trip_plan",
        "excluded_places",
        "included_places",
    ]:
        v = last_state.get(k)
        if v is not None:
            carry[k] = v

    return carry


def _reset_fields() -> Dict[str, Any]:
    """复刻你 main.py 每轮重置字段。"""
    return dict(
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


def run_one_turn(
    user_input: str,
    last_state: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], str]:
    """
    单轮对话：
    - 输入：user_input + 上一轮 last_state(dict)
    - 输出：final_state(dict) + final_response(str)
    """
    graph = get_graph()

    carry = _carry_from_last_state(last_state)
    reset = _reset_fields()

    initial_state = GraphState(**carry, **reset, user_input=user_input)

    final_state = None
    for event in graph.stream(initial_state, stream_mode="values"):
        final_state = event

        # 如果你想在 API 日志里看到节点流转，可打开下面这段（谨慎：日志会很多）
        # if config.debug and event.get("current_node"):
        #     print(f"[debug] node -> {event['current_node']}")

    if not final_state:
        return {}, "抱歉，处理过程中出现问题（empty final_state）"

    response = final_state.get("final_response") or "抱歉，处理过程中出现问题（empty final_response）"
    return final_state, response
