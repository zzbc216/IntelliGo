"""
LangGraph 图构建器
组装完整的工作流
"""
from langgraph.graph import StateGraph, END

from graph.state import GraphState
from graph.nodes import (
    node_rewrite,
    node_clarify_gate,
    node_intent_recognition,
    node_load_memory,
    node_fetch_weather,
    node_clothing_advice,
    node_trip_planning,
    node_risk_assessment,
    node_format_response,
    node_update_memory,
    node_general_qa,
)
from graph.edges import route_after_clarify, route_by_intent, should_replan


def build_graph() -> StateGraph:
    """
    构建 IntelliGo 工作流图

    流程:
    START -> intent_recognition -> load_memory -> fetch_weather
          -> [路由]
             ├─ clothing -> clothing_advice -> format_response
             ├─ planning -> trip_planning -> [风险检查]
             │                                ├─ replan -> risk_assessment -> format_response
             │                                └─ continue -> format_response
             ├─ general_qa -> general_qa -> format_response
             └─ general -> format_response
          -> update_memory -> END
    """
    # 创建图
    workflow = StateGraph(GraphState)

    # 添加节点
    workflow.add_node("rewrite", node_rewrite)
    workflow.add_node("intent_recognition", node_intent_recognition)
    workflow.add_node("clarify_gate", node_clarify_gate)
    workflow.add_node("load_memory", node_load_memory)
    workflow.add_node("fetch_weather", node_fetch_weather)
    workflow.add_node("clothing_advice", node_clothing_advice)
    workflow.add_node("trip_planning", node_trip_planning)
    workflow.add_node("risk_assessment", node_risk_assessment)
    workflow.add_node("general_qa", node_general_qa)
    workflow.add_node("format_response", node_format_response)
    workflow.add_node("update_memory", node_update_memory)

    # 设置入口
    workflow.set_entry_point("rewrite")

    # 添加边
    workflow.add_edge("rewrite", "intent_recognition")
    workflow.add_edge("intent_recognition", "clarify_gate")

    # 澄清门控：ask 直接格式化输出；continue 才继续后面的流程
    workflow.add_conditional_edges(
        "clarify_gate",
        route_after_clarify,
        {
            "ask": "update_memory",
            "continue": "load_memory",
        }
    )


    workflow.add_edge("load_memory", "fetch_weather")

    # 条件路由：根据意图分流
    workflow.add_conditional_edges(
        "fetch_weather",
        route_by_intent,
        {
            "clothing": "clothing_advice",
            "planning": "trip_planning",
            "general_qa": "general_qa",
            "general": "format_response"
        }
    )

    # 穿搭分支直接到格式化
    workflow.add_edge("clothing_advice", "format_response")

    # 通用问答分支直接到格式化
    workflow.add_edge("general_qa", "format_response")

    # 行程规划分支：检查是否需要重新规划
    workflow.add_conditional_edges(
        "trip_planning",
        should_replan,
        {
            "replan": "risk_assessment",
            "continue": "format_response"
        }
    )

    workflow.add_edge("risk_assessment", "format_response")

    # 格式化后更新记忆
    workflow.add_edge("format_response", "update_memory")

    # 结束
    workflow.add_edge("update_memory", END)

    return workflow


def get_compiled_graph():
    """获取编译后的图"""
    workflow = build_graph()
    return workflow.compile()


# 可视化图结构（调试用）
if __name__ == "__main__":
    graph = get_compiled_graph()

    # 打印图结构
    print("📊 IntelliGo 工作流图结构:")
    print(graph.get_graph().draw_ascii())
