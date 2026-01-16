"""
意图识别 + 实体抽取 Router
这是 IntelliGo 的"前台接待"
"""
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from utils.llm import get_structured_llm
from graph.state import UserIntent


class IntentExtractionResult(BaseModel):
    """LLM 结构化输出格式"""
    intent_type: str = Field(
        description="意图类型: clothing_advice(穿搭建议) / trip_planning(行程规划) / general_qa(旅行相关问答) / general_chat(闲聊) / unknown(无法识别)"
    )
    confidence: float = Field(ge=0, le=1, description="置信度 0-1")

    # 实体抽取
    cities: list[str] = Field(default_factory=list, description="提到的城市")
    dates: list[str] = Field(default_factory=list, description="提到的日期，格式 YYYY-MM-DD 或相对描述如'周末'")
    duration_days: int | None = Field(default=None, description="行程天数")
    preferences: list[str] = Field(default_factory=list, description="用户偏好关键词，如'安静'、'美食'、'拍照'")
    budget: str | None = Field(default=None, description="预算描述")
    excluded_places: list[str] = Field(default_factory=list, description="用户明确表示去过/不想去的地方，如'西湖去过了'中的'西湖'")
    included_places: list[str] = Field(default_factory=list, description="用户明确想去/还是想去的地方，如'我想去西湖'、'还是想去大别山'中的景点名")

    # general_qa 专用字段
    query_subject: str | None = Field(default=None, description="问答主题，如景点名、美食名、问题类型")
    has_health_concern: bool = Field(default=False, description="是否涉及健康相关问题")

    reasoning: str = Field(description="推理过程简述")


ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是 IntelliGo 的意图识别专家。分析用户输入，识别意图并抽取关键实体。

## 意图类型定义（按优先级排序）
1. **clothing_advice**: 用户询问穿什么、天气穿搭、出门穿衣建议、要带什么衣物/是否需要外套/冷不冷热不热
2. **trip_planning**: 用户想规划**完整行程**、旅游、周末去哪玩、多日游安排、路线/景点/时间表；也包括**修改/调整已有行程**的请求，如"换一个地方"、"我去过XX了"、"不想去XX"、"换个景点"等；**还包括补充行程相关信息**，如"时间改成XX"等
   - 注意：**单纯问餐厅/超市/电影院推荐不是 trip_planning**，而是 general_qa
3. **general_qa**: 旅行和生活服务相关的知识问答/推荐，包括：
   - 景点/地点详细介绍（如"介绍一下西湖"、"灵隐寺有什么特色"）
   - **餐厅/美食推荐**（如"有什么好吃的餐厅"、"推荐几家火锅店"、"预算100有哪些餐厅"）
   - **生活服务推荐**（如"附近有什么超市"、"哪个电影院比较好"、"有什么娱乐设施"、"KTV推荐"、"商场推荐"）
   - 交通相关问题（如"怎么去西湖"、"地铁方便吗"）
   - 健康出行问题（如"高血压能去爬山吗"、"糖尿病能吃这个吗"、"膝盖不好能走多久"）
   - 旅行常识（如"需要带什么证件"、"那边安全吗"）
   - **带预算的单项推荐**（如"预算1000有哪些热门餐厅"、"500块能去哪里玩"）—— 这类是**推荐问答**，不是行程规划
4. **general_chat**: 普通闲聊、问候、与旅行/穿搭完全无关的话题
5. **unknown**: 无法判断

## 意图判断优先级规则（非常重要）
- 只要用户明确在问"穿什么/带什么衣服/穿搭/要不要带外套/衣物清单/冷不冷热不热"，**无论是否同时提到去某城市玩几天**，都优先判定为 **clothing_advice**。
- **trip_planning 仅用于**：需要生成 day1-dayN 的完整行程安排、多景点路线规划、或修改已有行程
- **general_qa 用于**：
  - 单项推荐（餐厅、超市、电影院、娱乐设施、商场等）
  - 深入了解某个景点/美食/地点
  - 带预算的推荐问题（如"预算XX有什么好吃的"）
  - 典型触发词：推荐、有什么、哪家、哪个、介绍一下、怎么样、好吃吗、值得去吗
  - 健康相关：高血压、糖尿病、心脏病、膝盖、腰、孕妇、老人、小孩等 + 能不能/适合吗/可以吗
- 如果用户同时要"行程 + 穿搭"，优先选择用户句子中**问句的目标**：
  - 以"穿什么/带什么衣服"结尾或核心问题是衣物 => clothing_advice
  - 以"怎么安排/帮我规划/几天行程"结尾或核心问题是多日路线 => trip_planning

## 实体抽取规则
- 城市：识别所有提到的城市名
- 日期：转换为具体日期或保留相对描述（如"这周六"、"下周末"）
- 偏好：提取形容词或活动类型（如"安静的地方"、"想吃火锅"）
- 偏好 preferences：除了形容词，也要抽取"活动/场景/目的"，如 徒步/登山/拍照/逛街/看展/夜市/亲子/商务/通勤/泡温泉 等。
- 注意：不要把"衣服/穿搭/外套/穿什么"当成偏好 preferences，它们属于穿搭领域词。
- **排除景点 excluded_places**：提取用户明确说"去过了/不想去/换掉"的地方名称。
- **想去的景点 included_places**：提取用户明确说"想去/还是想去/加上XX/也去XX"的地方名称。
- **query_subject**：仅当 intent_type=general_qa 时提取，记录用户询问的主题（如"餐厅推荐"、"电影院"、"西湖"）
- **has_health_concern**：仅当 intent_type=general_qa 且涉及健康/疾病/身体状况时为 true

## 当前日期
{current_date}

## 示例（用于校准）
- 输入：我想去杭州玩3天，我应该带什么衣服？
  输出：intent_type=clothing_advice，cities=["杭州"], duration_days=3
- 输入：周末去杭州玩两天，喜欢安静的地方
  输出：intent_type=trip_planning，cities=["杭州"], duration_days=2, preferences包含"安静"
- 输入：明天北京穿什么合适？
  输出：intent_type=clothing_advice，cities=["北京"], dates=["明天"]
- 输入：我已经去过西湖了，换一个地方
  输出：intent_type=trip_planning（这是修改行程的请求）
- 输入：仔细介绍一下西湖
  输出：intent_type=general_qa，query_subject="西湖"
- 输入：高血压能去爬山吗
  输出：intent_type=general_qa，query_subject="高血压爬山"，has_health_concern=true
- 输入：杭州有什么好吃的
  输出：intent_type=general_qa，cities=["杭州"]，query_subject="杭州美食"
- 输入：预算1000，有哪些热门餐厅值得一试
  输出：intent_type=general_qa，query_subject="热门餐厅推荐"，budget="1000"（这是餐厅推荐问答，不是行程规划）
- 输入：武汉有什么好吃的餐厅
  输出：intent_type=general_qa，cities=["武汉"]，query_subject="武汉餐厅推荐"
- 输入：附近有什么超市
  输出：intent_type=general_qa，query_subject="超市推荐"
- 输入：推荐几家电影院
  输出：intent_type=general_qa，query_subject="电影院推荐"
- 输入：有什么娱乐设施
  输出：intent_type=general_qa，query_subject="娱乐设施推荐"
- 输入：哪个商场比较好逛
  输出：intent_type=general_qa，query_subject="商场推荐"
- 输入：KTV有推荐的吗
  输出：intent_type=general_qa，query_subject="KTV推荐"

请仔细分析，给出结构化结果。"""),
    ("human", "{user_input}")
])


class IntentRouter:
    """意图路由器"""

    def __init__(self):
        self.chain = ROUTER_PROMPT | get_structured_llm(IntentExtractionResult, temperature=0.0)

    def analyze(self, user_input: str, current_date: str) -> UserIntent:
        """分析用户输入，返回意图和实体"""
        result: IntentExtractionResult = self.chain.invoke({
            "user_input": user_input,
            "current_date": current_date
        })

        return UserIntent(
            intent_type=result.intent_type,
            confidence=result.confidence,
            extracted_entities={
                "cities": result.cities,
                "dates": result.dates,
                "duration_days": result.duration_days,
                "preferences": result.preferences,
                "budget": result.budget,
                "excluded_places": result.excluded_places,
                "included_places": result.included_places,
                "query_subject": result.query_subject,
                "has_health_concern": result.has_health_concern,
                "reasoning": result.reasoning
            }
        )


# 单例
router = IntentRouter()

# ---- 测试代码 ----
if __name__ == "__main__":
    from datetime import date

    test_inputs = [
        "周末想去杭州玩两天，喜欢安静的地方",
        "明天北京穿什么合适？",
        "帮我规划一下下周从上海到苏州的三日游，预算2000左右",
        "你好呀",
    ]

    for inp in test_inputs:
        print(f"\n📝 输入: {inp}")
        result = router.analyze(inp, str(date.today()))
        print(f"🎯 意图: {result.intent_type} (置信度: {result.confidence})")
        print(f"📦 实体: {result.extracted_entities}")
