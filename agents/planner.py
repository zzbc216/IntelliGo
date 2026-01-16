"""
行程规划 Agent
支持多城市、多日行程编排

给定用户需求（去哪里、玩几天、偏好、预算）+ 已有信息（天气、用户画像），让大模型生成一个可用的、多天多城市行程，并把结果转换成你系统内部统一使用的格式
"""
from pydantic import BaseModel, Field,ConfigDict
from langchain_core.prompts import ChatPromptTemplate
from utils.llm import get_structured_llm
from graph.state import TripPlan, TripDay, WeatherInfo

#  规定“模型必须按什么格式回答”

class PlannerActivity(BaseModel):
    model_config = ConfigDict(extra="forbid")            #开启「严格模式」，禁止出现任何未定义的字段
    time: str = Field(description="如 09:00")
    name: str = Field(description="活动/景点名")
    description: str = Field(description="简介")
    duration: str = Field(description="如 2小时")
    cost: str = Field(description="如 50元/免费")

class PlannerDay(BaseModel):
    model_config = ConfigDict(extra="forbid")
    date: str = Field(description="日期，允许相对描述或 YYYY-MM-DD")
    city: str = Field(description="城市")
    activities: list[PlannerActivity] = Field(description="当天活动列表",min_length = 1)

class PlannerOutput(BaseModel):
    """规划器输出格式"""
    model_config = ConfigDict(extra="forbid")
    title: str = Field(description="行程标题")
    days: list[PlannerDay] = Field(description="每日行程")
    total_budget_estimate: str = Field(description="总预算估计")
    tips: list[str] = Field(description="旅行小贴士")


PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是专业的旅行规划师，擅长设计个性化行程。

## 规划原则
1. **节奏合理**：每天 3-4 个主要活动，避免过于紧凑
2. **动线优化**：相近景点安排在同一天
3. **用户优先**：根据用户偏好调整推荐
4. **天气适配**：雨天推荐室内活动，晴天安排户外
5. **本地特色**：融入当地美食和文化体验

## 预算使用规则（非常重要！！！）
用户预算是他们**愿意花费的金额**，你的方案应该**充分利用预算**，而不是尽量省钱！

### 预算匹配原则：
- 如果用户预算是 1000 元，你的方案总花费应该在 **800-1000 元** 左右
- 如果用户预算是 500 元，你的方案总花费应该在 **400-500 元** 左右
- 如果用户预算是 2000 元，你的方案总花费应该在 **1600-2000 元** 左右
- **永远不要规划出远低于预算的方案**（如预算1000却只花250是错误的！）

### 如何用好预算：
- **高预算（>1000元/天）**：
  - 米其林/黑珍珠餐厅、网红餐厅（人均 150-300 元）
  - 付费景点 VIP 体验、私人导览（100-300 元）
  - 特色体验（游船、表演、手作课程等 100-200 元）
  - 打车/专车出行

- **中等预算（500-1000元/天）**：
  - 特色餐厅（人均 80-150 元）
  - 主要付费景点门票（50-150 元）
  - 1-2 个特色体验项目
  - 地铁+偶尔打车

- **低预算（<500元/天）**：
  - 当地小吃、平价餐厅（人均 30-60 元）
  - 免费或低价景点为主
  - 公共交通

### 每个活动必须标注费用：
- 门票费用（如：西湖免费、灵隐寺75元）
- 餐饮费用（如：人均80元、约150元）
- 体验费用（如：游船50元、演出180元）
- 交通费用（如：打车约30元）

## 修改行程规则（非常重要！）
如果用户提供了"之前的行程"并要求修改：
- **必须仔细阅读用户的修改请求**
- 如果用户说"XX去过了"、"不想去XX"、"换掉XX"，**必须完全移除**该景点，用其他景点替代
- 即使用户只说了部分名称（如"大别山"），也要排除包含该关键词的所有景点（如"黄冈大别山"）
- 如果用户调整预算（更高/更低），**必须相应调整推荐的档次和总花费**
- 推荐同城市、同类型的替代景点
- 保持行程的其他部分不变（除非用户要求全部重新规划）

## 输出格式
每日行程包含:
- date: 日期
- city: 城市
- activities: [
    {{"time": "09:00", "name": "景点名", "description": "简介", "duration": "2小时", "cost": "门票50元"}}
  ]

**total_budget_estimate 必须是所有活动费用的合计，格式如："约850元（门票200+餐饮400+交通100+体验150）"**

请根据用户需求生成详细行程，确保总花费接近用户预算！"""),
    ("human", """## 用户当前请求（最重要！必须响应这个请求！）
{user_input}

## ⚠️ 调整指令（如果有，必须严格执行！）
{adjustment_hint}

## 提取的信息
- 目的地城市: {cities}
- 日期/天数: {dates}
- 偏好: {preferences}
- 用户预算: {budget}（⚠️ 你的方案总花费应接近这个数字！）

## 必须排除的景点（绝对不能出现！）
{excluded_places}

## 必须包含的景点（尽量安排进行程）
{included_places}

## 天气预报
{weather_info}

## 用户画像
{user_profile}

## 之前的行程（仅供参考，如果有调整指令，必须生成不同的行程！）
{previous_plan}

请根据用户请求生成行程规划，**确保总花费接近用户预算 {budget}**！""")
])


class TripPlanner:
    """行程规划器"""

    def __init__(self):
        self.chain = PLANNER_PROMPT | get_structured_llm(PlannerOutput, temperature=0.3)

    def plan(self, context: dict) -> TripPlan:
        """生成行程规划"""
        entities = context.get("entities", {})
        weather_data = context.get("weather_data", {})
        user_profile = context.get("user_profile", {})

        # 格式化天气信息（兼容：dict / WeatherInfo / list[dict|WeatherInfo]）
        weather_info = ""
        for city, weather in (weather_data or {}).items():
            if weather is None:
                weather_info += f"- {city}: 暂无天气数据\n"
                continue

            # 多日预报
            if isinstance(weather, list):
                weather_info += f"- {city}（多日预报）:\n"
                for i, w in enumerate(weather, start=1):
                    if isinstance(w, dict):
                        weather_info += f"  - D{i}: {w.get('weather', '未知')} {w.get('temperature', '?')}°C\n"
                    else:
                        weather_info += f"  - D{i}: {w.weather} {w.temperature}°C\n"
                continue

            # 单日
            if isinstance(weather, dict):
                weather_info += f"- {city}: {weather.get('weather', '未知')} {weather.get('temperature', '?')}°C\n"
            else:
                weather_info += f"- {city}: {weather.weather} {weather.temperature}°C\n"

        # 格式化用户画像
        profile_str = ""
        if user_profile.get("relevant_memories"):
            memories = [m["content"] for m in user_profile["relevant_memories"]]
            profile_str = "相关偏好: " + ", ".join(memories)

        # 格式化之前的行程（用于修改请求）
        previous_plan = context.get("previous_plan")
        previous_plan_str = "无（这是新的行程规划请求）"
        if previous_plan and getattr(previous_plan, "days", None):
            lines = [f"标题: {previous_plan.title}"]
            for i, day in enumerate(previous_plan.days, 1):
                day_activities = []
                for act in (day.activities or []):
                    name = act.get("name", "") if isinstance(act, dict) else getattr(act, "name", "")
                    if name:
                        day_activities.append(name)
                lines.append(f"Day {i} ({day.city}): {', '.join(day_activities) or '无活动'}")
            previous_plan_str = "\n".join(lines)

        # 格式化排除和包含的景点
        excluded_places = context.get("excluded_places") or []
        excluded_str = ", ".join(excluded_places) if excluded_places else "无"

        included_places = context.get("included_places") or []
        included_str = ", ".join(included_places) if included_places else "无"

        # 调用 LLM
        adjustment_hint = context.get("adjustment_hint") or "无特殊调整要求"
        result = self.chain.invoke({
            "user_input": context.get("user_input", ""),
            "adjustment_hint": adjustment_hint,
            "cities": ", ".join(entities.get("cities", ["未指定"])),
            "dates": ", ".join(entities.get("dates", [])) or f"{entities.get('duration_days', 2)}天",
            "preferences": ", ".join(entities.get("preferences", ["无特别偏好"])),
            "budget": entities.get("budget") or "未指定",
            "excluded_places": excluded_str,
            "included_places": included_str,
            "weather_info": weather_info or "暂无天气数据",
            "user_profile": profile_str or "新用户，暂无历史偏好",
            "previous_plan": previous_plan_str
        })

        # 转换为 TripPlan
        days = []
        for idx, day_data in enumerate(result.days):
            city = getattr(day_data, "city", "")
            date = getattr(day_data, "date", "")

            weather = None
            if city in (weather_data or {}):
                w = weather_data[city]

                # ✅ 多日：按 idx 取对应天，超出则取最后一天兜底
                if isinstance(w, list) and w:
                    pick = w[min(idx, len(w) - 1)]
                    if isinstance(pick, dict):
                        weather = WeatherInfo(**pick)
                    else:
                        weather = pick

                # 单日：dict / WeatherInfo
                elif isinstance(w, dict):
                    weather = WeatherInfo(**w)
                else:
                    weather = w  # WeatherInfo 或 None

            activities = []
            for a in getattr(day_data, "activities", []) or []:
                activities.append(a.model_dump() if hasattr(a, "model_dump") else a.dict())

            days.append(TripDay(
                date=date,
                city=city,
                activities=activities,
                weather=weather
            ))

        return TripPlan(
            title=result.title,
            days=days,
            total_budget_estimate=result.total_budget_estimate,
            tips=result.tips
        )

    def generate_backup(self, day: TripDay) -> str:
        """为特定日期生成备选方案"""
        from utils.llm import get_llm

        llm = get_llm(temperature=0.5)
        prompt = f"""原计划在 {day.city} 的行程因天气({day.weather.weather if day.weather else '恶劣'})可能受影响。
原活动: {[a.get('name') for a in day.activities]}

请推荐 2-3 个室内替代方案（博物馆、商场、室内景点等），用一句话概括。"""

        response = llm.invoke(prompt)
        return response.content


# 单例
trip_planner = TripPlanner()

# ---- 测试 ----
if __name__ == "__main__":
    context = {
        "user_input": "周末想去杭州玩两天，喜欢安静的地方",
        "entities": {
            "cities": ["杭州"],
            "dates": ["周六", "周日"],
            "duration_days": 2,
            "preferences": ["安静"],
            "budget": None
        },
        "weather_data": {
            "杭州": {"city": "杭州", "weather": "多云", "temperature": 18}
        },
        "user_profile": {}
    }

    plan = trip_planner.plan(context)
    print(f"📋 {plan.title}")
    for day in plan.days:
        print(f"\n📅 {day.date} - {day.city}")
        for act in day.activities:
            print(f"  - {act.get('time')} {act.get('name')}")
