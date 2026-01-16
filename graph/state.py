"""
LangGraph 状态定义 - IntelliGo 的"大脑记忆"
"""
from typing import Annotated, Literal
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages


class UserIntent(BaseModel):
    """用户意图识别结果"""
    intent_type: Literal["clothing_advice", "trip_planning", "general_qa", "general_chat", "unknown"] = Field(
        description="识别出的意图类型"
    )
    confidence: float = Field(ge=0, le=1, description="置信度")
    extracted_entities: dict = Field(default_factory=dict, description="抽取的实体")


class WeatherInfo(BaseModel):
    """天气信息"""
    city: str = ""
    temperature: float = 0.0
    weather: str = ""  # 晴/多云/雨...
    humidity: int = 0
    wind_power: str = ""
    suggestion: str = ""  # 天气建议
    raw_data: dict = Field(default_factory=dict)


class TripDay(BaseModel):
    """单日行程"""
    date: str = ""
    city: str = ""
    activities: list[dict] = Field(default_factory=list)
    weather: WeatherInfo | None = None
    risk_level: Literal["low", "medium", "high"] = "low"
    backup_plan: str = ""


class TripPlan(BaseModel):
    """完整行程规划"""
    title: str = ""
    days: list[TripDay] = Field(default_factory=list)
    total_budget_estimate: str = ""
    tips: list[str] = Field(default_factory=list)


class GraphState(BaseModel):
    """
    LangGraph 主状态 - 贯穿整个工作流的上下文

    使用 Pydantic v2 以获得更好的类型检查和序列化
    """
    # 对话历史 (使用 LangGraph 的消息累加器)
    messages: Annotated[list, add_messages] = Field(default_factory=list)

    # 当前用户输入
    user_input: str = ""

    # ===== rewrite / 澄清相关 =====
    rewritten_query: str = ""
    rewrite_slots: dict = Field(default_factory=dict)
    duration_days_is_default: bool = False

    need_clarification: bool = False
    clarifying_questions: list[str] = Field(default_factory=list)

    # 是否本轮仅做澄清提问（缺城市时 True：直接输出问题，不继续规划）
    clarify_only: bool = False

    # 意图识别结果
    intent: UserIntent | None = None

    # 抽取的关键实体
    entities: dict = Field(default_factory=dict)
    # 示例: {"cities": ["北京", "上海"], "dates": ["2024-01-20", "2024-01-22"], "preferences": ["安静", "咖啡"]}

    # 用户排除的景点（跨轮继承）
    excluded_places: list[str] = Field(default_factory=list)

    # 用户想去的景点（跨轮继承，优先级高于排除）
    included_places: list[str] = Field(default_factory=list)

    # 天气数据缓存
    weather_data: dict[str, WeatherInfo | list[WeatherInfo]] = Field(default_factory=dict)  # city -> WeatherInfo 或 多日预报列表

    # 用户画像 (从记忆系统加载)
    user_profile: dict = Field(default_factory=dict)

    # 行程规划结果
    trip_plan: TripPlan | None = None

    # 穿搭建议
    clothing_advice: str = ""

    # 最终输出
    final_response: str = ""

    # 流程控制
    current_node: str = ""
    needs_replan: bool = False
    error_message: str = ""
