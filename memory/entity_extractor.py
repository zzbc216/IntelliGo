"""
从对话中抽取用户偏好实体
用于自动更新用户画像
"""

from pydantic import BaseModel, Field,ConfigDict
from langchain_core.prompts import ChatPromptTemplate
from utils.llm import get_structured_llm

class PreferenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: str = Field(description="偏好内容")
    category: str = Field(description="偏好类别，如 dining/activity/travel_style/budget/accommodation/health")

class ExtractedPreferences(BaseModel):
    """抽取的偏好"""
    model_config = ConfigDict(extra="forbid")
    preferences: list[PreferenceItem] = Field(
        default_factory=list,
        description="抽取的偏好列表"
    )
    has_new_info: bool = Field(description="是否包含新的用户偏好信息")


EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是用户画像分析专家。从对话中提取用户的**持久性偏好**（不是临时需求）。

## 类别定义
- **dining**: 餐饮偏好（口味、餐厅类型、忌口）
- **activity**: 活动偏好（喜欢/不喜欢的活动类型）
- **travel_style**: 旅行风格（节奏、人群、景点类型）
- **budget**: 预算偏好
- **accommodation**: 住宿偏好
- **health**: 健康相关限制

## 抽取规则
1. 只提取**明确表达的偏好**，不要推测
2. 偏好应该是**可复用的**（下次还适用）
3. 负面偏好也要记录（如"不喜欢辣"）
4. 忽略临时性需求（如"这次想吃火锅"不算偏好）

## 示例
用户说："我膝盖不太好，爬山有点吃力，而且我是素食主义者"
抽取：
- {{"content": "膝盖不好，不适合爬山等高强度活动", "category": "health"}}
- {{"content": "素食主义者", "category": "dining"}}"""),
    ("human", """## 对话内容
{conversation}

请分析并抽取用户偏好：""")
])


class EntityExtractor:
    """实体/偏好抽取器"""

    def __init__(self):
        self.chain = EXTRACTION_PROMPT | get_structured_llm(ExtractedPreferences, temperature=0.0)

    def extract(self, conversation: str) -> ExtractedPreferences:
        """从对话中抽取偏好"""
        return self.chain.invoke({"conversation": conversation})


# 单例
entity_extractor = EntityExtractor()

# ---- 测试 ----
if __name__ == "__main__":
    test_conversation = """
    用户: 帮我规划周末杭州两日游
    助手: 好的，请问有什么特别的偏好吗？
    用户: 我不太喜欢人多的地方，最好是文艺一点的。对了我吃素，帮我推荐一些素食餐厅。还有我住酒店比较挑，至少要四星级的。
    """

    result = entity_extractor.extract(test_conversation)
    print(f"🔍 发现新偏好: {result.has_new_info}")
    for pref in result.preferences:
        print(f"  - [{pref.category}] {pref.content}")