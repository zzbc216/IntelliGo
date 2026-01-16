"""
LangGraph 条件边定义
决定流程走向
"""
from typing import Literal
from graph.state import GraphState

def route_after_clarify(state: GraphState) -> Literal["ask", "continue"]:
    """
    澄清门控后的路由：
    - ask: 本轮仅提问澄清（缺城市），直接走 format_response
    - continue: 信息足够，继续 load_memory -> fetch_weather -> ...
    """
    return "ask" if state.clarify_only else "continue"

def route_by_intent(state: GraphState) -> Literal["clothing", "planning", "general_qa", "general"]:
    """
    根据意图路由到不同分支
    优先级: clothing_advice > trip_planning > general_qa > general_chat/unknown
    """
    if not state.intent:
        return "general"

    intent_type = state.intent.intent_type

    if intent_type == "clothing_advice":
        return "clothing"
    elif intent_type == "trip_planning":
        return "planning"
    elif intent_type == "general_qa":
        return "general_qa"
    else:
        return "general"


def should_replan(state: GraphState) -> Literal["replan", "continue"]:
    """
    判断是否需要重新规划
    """
    if state.needs_replan:
        return "replan"
    return "continue"
