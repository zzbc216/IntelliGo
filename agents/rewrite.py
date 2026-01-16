"""
Rewrite 节点：把用户输入规范化成更可执行的查询，并抽取关键槽位、生成最少澄清问题
不做时间解析：例如“这周末”原样保留在 dates_text
"""
from __future__ import annotations
from datetime import date
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field,ConfigDict
from langchain_core.prompts import ChatPromptTemplate
from utils.llm import get_structured_llm

class RewriteSlots(BaseModel):
    # 关键：禁止额外字段 -> JSON Schema additionalProperties=false
    model_config = ConfigDict(extra="forbid")

    cities: List[str] = Field(default_factory=list, description="提到的城市/目的地列表")
    duration_days: Optional[int] = Field(default=None, description="行程天数；缺失则为 null（默认策略在下游处理）")
    preferences: List[str] = Field(default_factory=list, description="偏好关键词，如 美食/轻松/拍照/安静")
    budget_text: Optional[str] = Field(default=None, description="预算原话，如 别太贵/人均500/预算充足")
    dates_text: Optional[str] = Field(default=None, description="日期原话，如 这周末/下周/国庆（不做时间解析）")

class RewriteResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rewritten_query: str = Field(..., description="不改变含义的清晰改写，用于后续路由/检索/规划")
    slots: RewriteSlots = Field(default_factory=RewriteSlots, description="抽取的关键槽位（固定字段）")
    need_clarification: bool = Field(..., description="是否需要澄清才能继续高质量执行")
    clarifying_questions: List[str] = Field(default_factory=list, description="关键追问，1~3个，尽量合并并给可选项")
    confidence: float = Field(..., ge=0.0, le=1.0, description="置信度")
    notes: Optional[str] = None


REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个对话请求规范化（rewrite）与缺口识别助手。\n"
            "目标：不改变用户意图，把用户话改写为更清晰、可执行的请求，并抽取关键槽位。\n"
            "重要约束：\n"
            "1) 不要做时间解析：例如不要把“这周末”转换为具体日期；原样保留即可。\n"
            "2) 不要编造用户未提供的信息；缺失就标记缺失并生成澄清问题。\n"
            "3) 澄清问题要少而关键（1~2个优先），并给出选项或可直接填写格式。\n"
            "澄清判定（仅针对出行/行程规划类请求）：\n"
             "若 cities 为空：必须追问“目的地城市/区域。\n"
             "若 duration_days 为空：优先追问“玩几天（给 1/2/3 天选项）。\n"
             "若 dates_text 为空：次优先追问“什么时候去（如这周末/下周/某日期）。\n"
             " 预算 budget_text 可选：仅在上述两项都不缺时再追问，或用户明显在意价格时追问\n"
             " 澄清问题数量上限：最多 2 个，能合并就合并。\n"
              "当前日期：{current_date}\n",
        ),
        (
            "human",
            "用户输入：{user_input}\n\n"
            "请输出结构化结果。slots 建议包含：\n"
            "- cities: list[str]\n"
            "- duration_days: int | null\n"
            "- preferences: list[str]\n"
            "- budget_text: str | null（把用户所有预算相关原话合并保留，例如“别太贵/经济实惠/人均500”）\n"
            "- dates_text: str | null（如“这周末/下周/国庆”，保持原样）\n",
        ),
    ]
)


class Rewriter:
    def __init__(self):
        self.llm = get_structured_llm(RewriteResult, temperature=0.0)

    def rewrite(self, user_input: str, current_date: Optional[str] = None) -> RewriteResult:
        if not current_date:
            current_date = str(date.today())
        chain = REWRITE_PROMPT | self.llm
        return chain.invoke({"user_input": user_input, "current_date": current_date})


rewriter = Rewriter()

# ---- 小测试代码：直接运行本文件即可 ----
def _pretty_print(result: RewriteResult):
    print("\n[rewritten_query]")
    print(result.rewritten_query)

    print("\n[slots]")
    slots = result.slots.model_dump() if result.slots else {}
    for k in ["cities", "duration_days", "preferences", "budget_text", "dates_text"]:
        if k in slots:
            print(f"- {k}: {slots.get(k)}")
    # 打印其它 slots（如果模型返回了额外字段）
    extra_keys = [k for k in slots.keys() if k not in {"cities", "duration_days", "preferences", "budget_text", "dates_text"}]
    for k in extra_keys:
        print(f"- {k}: {slots.get(k)}")

    print("\n[need_clarification]")
    print(result.need_clarification)

    print("\n[clarifying_questions]")
    if result.clarifying_questions:
        for q in result.clarifying_questions:
            print(f"- {q}")
    else:
        print("- (none)")

    print("\n[confidence]")
    print(result.confidence)

    if result.notes:
        print("\n[notes]")
        print(result.notes)


if __name__ == "__main__":
    cases = [
        # 1) 缺城市：应该要求目的地
        "周末想出去玩，帮我安排一下",
        # 2) 有城市、缺天数：应追问几天（不在这里默认 1 天游，默认策略在 nodes 里做）
        "去武汉玩，想轻松一点，美食多一点",
        # 3) 有预算表达：budget_text 应抓到原话
        "下周去成都玩两天，人均2000左右，想住舒服点但别太折腾",
        # 4) 只说预算偏好：也应能抽到 budget_text
        "尽量别太贵，经济实惠一点",
    ]

    for i, text in enumerate(cases, 1):
        print("\n" + "=" * 80)
        print(f"CASE {i} USER: {text}")
        try:
            r = rewriter.rewrite(text)
            _pretty_print(r)
        except Exception as e:
            print("ERROR:", repr(e))
