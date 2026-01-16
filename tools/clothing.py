"""
穿搭建议生成器
结合天气数据给出科学的穿衣建议（支持“配套穿搭/套装化输出”）
"""
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from utils.llm import get_structured_llm
from graph.state import WeatherInfo


class ClothingAdvice(BaseModel):
    """穿搭建议结构（方案 B：套装化）"""
    summary: str = Field(description="一句话总结（要体现活动/场景与温度）")

    # 配套穿搭关键字段
    layers: str = Field(description="分层/外套策略（例如：内搭+中层+外层，以及是否可脱穿）")
    shoes: str = Field(description="鞋子建议（步行强度/雨天防滑/正式度）")
    outfit_set: str = Field(description="一套可直接照抄的穿搭组合（从上到下，含鞋/外套/配件）")

    top: str = Field(description="上衣建议")
    bottom: str = Field(description="下装建议")
    accessories: list[str] = Field(description="配件建议，如帽子、围巾、伞等")
    tips: list[str] = Field(description="额外提示（结合活动风险：走路磨脚、出汗、早晚温差等）")
    confidence: str = Field(description="建议可信度说明（基于天气信息完整度/活动信息完整度）")


CLOTHING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是专业的穿搭顾问，根据天气数据给出【可直接照抄的配套穿搭】。

## 穿搭原则
1. **温度分层**：
   - <5°C: 羽绒服/厚棉服 + 毛衣 + 保暖内衣
   - 5-15°C: 外套/风衣 + 卫衣/薄毛衣
   - 15-22°C: 薄外套/衬衫 + 长袖T恤
   - 22-28°C: 短袖/薄长袖
   - >28°C: 短袖短裤，注意防晒

2. **天气适配**：
   - 雨天：防水外套，避免浅色鞋/麂皮鞋，建议防滑
   - 大风：注意防风，裙装/阔腿裤需考虑风，优先贴身或可收口
   - 晴热：防晒衣/帽子/墨镜，注意补水

3. **场景考虑**：
   - 必须根据用户描述的活动/场景调整：正式度、可活动性、耐走、是否出汗、是否需要拍照好看

## 输出要求（非常重要）
- 你必须输出：
  - layers（分层/外套策略）
  - shoes（鞋子建议）
  - outfit_set（一整套从上到下可照抄的搭配，含鞋/外套/配件）
- 若用户活动/场景信息不足：
  - 仍给出一个“默认日常出行”的 outfit_set
  - 同时在 tips 里给出 1 条“需要补充的活动问题”（例如徒步/拍照/正式场合/夜间活动）
## 输出硬性要求（必须遵守）
1. 你的输出会被结构化解析为 ClothingAdvice。
2. summary 必须使用以下句式（必须包含“今天要做什么/去哪里”与“温度/体感”）：
   - “考虑到你今天要【活动/地点】（【活动标签】），结合当日【白天xx°C/夜间xx°C + 天气要点】，所以建议：……”
3. tips 至少包含 2 条，且要与当天活动相关（例如：爬山出汗、久走磨脚、寺庙需端庄、夜间降温、雨天防滑）。
4. shoes 必须结合“步行/上下坡/雨天防滑/正式度”给理由。

请给出具体、可执行的建议，避免泛泛而谈。"""),
    ("human", """## 天气信息
- 城市：{city}
- 天气：{weather}
- 温度：{temperature}°C
- 湿度：{humidity}%
- 风力：{wind_power}

## 用户补充（包含活动/场景/偏好/原话/记忆）
{user_context}

请给出今日穿搭建议："""),
])


class ClothingAdvisor:
    """穿搭顾问"""

    def __init__(self):
        # 结构化输出：ClothingAdvice
        self.chain = CLOTHING_PROMPT | get_structured_llm(ClothingAdvice, temperature=0.3)

    def advise(self, weather: WeatherInfo, user_context: str = "") -> ClothingAdvice:
        """生成穿搭建议"""
        return self.chain.invoke({
            "city": weather.city,
            "weather": weather.weather,
            "temperature": weather.temperature,
            "humidity": weather.humidity,
            "wind_power": weather.wind_power or "微风",
            "user_context": user_context or "日常出行",
        })

    def format_advice(self, advice: ClothingAdvice) -> str:
        """格式化输出"""
        lines = [
            "👔 **今日穿搭建议**",
            "",
            f"📝 {advice.summary}",
            "",
            f"**分层/外套策略**: {advice.layers}",
            f"**上衣**: {advice.top}",
            f"**下装**: {advice.bottom}",
            f"**鞋子**: {advice.shoes}",
            "",
            f"**一套照抄（Outfit）**: {advice.outfit_set}",
        ]

        if advice.accessories:
            lines.append(f"**配件**: {', '.join(advice.accessories)}")

        if advice.tips:
            lines.append("")
            lines.append("💡 **小贴士**:")
            for tip in advice.tips:
                lines.append(f"  - {tip}")

        # 你也可以把 confidence 打出来（可选）
        if advice.confidence:
            lines.append("")
            lines.append(f"**可信度**: {advice.confidence}")

        return "\n".join(lines)


# 单例
clothing_advisor = ClothingAdvisor()


# ---- 测试 ----
if __name__ == "__main__":
    from tools.weather import weather_tool

    weather = weather_tool.get_weather("北京")
    print(f"🌤️ 天气: {weather.city} {weather.weather} {weather.temperature}°C\n")

    advice = clothing_advisor.advise(weather, "今天要去面试")
    print(clothing_advisor.format_advice(advice))
