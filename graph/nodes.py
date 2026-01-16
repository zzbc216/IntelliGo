"""
LangGraph 节点定义

"""
from datetime import date
from langchain_core.messages import AIMessage, HumanMessage
from graph.state import GraphState, TripPlan, TripDay
from agents.router import router
from tools.weather import weather_tool
from tools.clothing import clothing_advisor
from memory.vector_store import UserMemory
from memory.entity_extractor import entity_extractor
from agents.rewrite import rewriter

# 初始化记忆系统
user_memory = UserMemory()


from typing import Any


def _extract_day_activity_names(day: Any) -> list[str]:
    """
    TripDay.activities: list[dict] -> 提取活动名称
    兼容常见 key：name/title/place/spot/activity
    """
    names: list[str] = []
    activities = getattr(day, "activities", None) or []

    for a in activities:
        if not isinstance(a, dict):
            continue

        # ✅ 优先 name，其它 key 作为兼容
        for key in ("name", "title", "place", "spot", "activity"):
            v = a.get(key)
            if isinstance(v, str) and v.strip():
                names.append(v.strip())
                break

    # 去重保序
    return list(dict.fromkeys(names))


def _infer_activity_tags(activity_names: list[str]) -> list[str]:
    """
    从活动/景点名称粗略推断穿搭相关标签，用于让 LLM 更“按场景说话”
    """
    text = " ".join(activity_names)
    tags: list[str] = []

    def hit(words: list[str]) -> bool:
        return any(w in text for w in words)

    if hit(["山", "岭", "峰", "徒步", "登高", "爬"]):
        tags.append("爬山/徒步（出汗+风大+上下坡）")
    if hit(["湿地", "公园", "植物园", "湖", "江", "西湖", "钱塘江", "游船"]):
        tags.append("户外长时间（防风/防晒/耐走）")
    if hit(["博物馆", "美术馆", "展", "大剧院", "剧院"]):
        tags.append("室内为主（温差/空调/体面）")
    if hit(["寺", "庙", "灵隐", "祠"]):
        tags.append("寺庙/人文（端庄+好走）")
    if hit(["夜景", "夜游", "演出", "千古情", "秀"]):
        tags.append("夜间活动（降温+拍照）")
    if hit(["古街", "街区", "citywalk", "暴走", "打卡", "拍照"]):
        tags.append("Citywalk/拍照（轻便+上镜）")

    if not tags:
        tags.append("城市休闲（通用步行）")

    return list(dict.fromkeys(tags))



def node_rewrite(state: GraphState) -> dict:
    """
    节点0: rewrite（规范化 + 槽位抽取 + 缺口识别）
    策略：
    - 不做时间解析
    - 默认天数为 1 天游（缺失时）
    - 城市缺失：后续由 clarify_gate 强制追问并中止本轮
    """
    r = rewriter.rewrite(state.user_input)

    if r.slots is None:
        slots = {}
    elif hasattr(r.slots, "model_dump"):
        slots = r.slots.model_dump()
    else:
        slots = dict(r.slots)

    # 确保字段存在且类型友好
    if not isinstance(slots.get("cities"), list):
        slots["cities"] = slots.get("cities") or []
    if not isinstance(slots.get("preferences"), list):
        slots["preferences"] = slots.get("preferences") or []

    # 默认天数策略：缺失则设为 1（你指定）
    duration_days_is_default = False
    if slots.get("duration_days") in [None, "", 0]:
        duration_days_is_default = True
        slots["duration_days"] = 1
        # 仍给一个轻量追问（不拦截）
        qs = list(r.clarifying_questions or [])
        if not any(("几天" in q) or ("天" in q) for q in qs):
            qs.insert(0, "你这次预计玩几天（1/2/3天）？我先按 1 天游给你一个初稿。")
        clarifying_questions = qs[:3]
    else:
        clarifying_questions = list(r.clarifying_questions or [])[:3]

    return {
        "rewritten_query": r.rewritten_query,
        "rewrite_slots": slots,
        "need_clarification": r.need_clarification,
        "clarifying_questions": clarifying_questions,
        "duration_days_is_default": duration_days_is_default,
        "current_node": "rewrite",
    }


def node_intent_recognition(state: GraphState) -> dict:
    """
    节点1: 意图识别 + 实体抽取

    ✅ 修复点：
    - 不要每轮整包覆盖 entities
    - 本轮没抽到 cities 时，沿用历史 state.entities["cities"]
    - preferences 做并集，duration_days 有新值就覆盖
    """
    user_input = state.rewritten_query or state.user_input
    current_date = str(date.today())

    # 调用 Router 分析（本轮抽取）
    intent = router.analyze(user_input, current_date)

    new_entities = dict(intent.extracted_entities or {})
    rs = state.rewrite_slots or {}

    # 用 rewrite_slots 补强本轮抽取
    if rs.get("cities") and not new_entities.get("cities"):
        new_entities["cities"] = rs["cities"]

    if (
            rs.get("duration_days")
            and not new_entities.get("duration_days")
            and not getattr(state, "duration_days_is_default", False)
    ):
        new_entities["duration_days"] = rs["duration_days"]

    if rs.get("preferences") and not new_entities.get("preferences"):
        new_entities["preferences"] = rs["preferences"]

    # budget 先原样放着（不强制数值化）
    if rs.get("budget_text") and not new_entities.get("budget"):
        new_entities["budget"] = rs["budget_text"]
    if rs.get("dates_text") and not new_entities.get("dates"):
        new_entities["dates"] = [rs["dates_text"]]

    # ✅ 与历史 entities 合并（保留“已确认信息”）
    merged = dict(state.entities or {})

    # cities：本轮有就覆盖，本轮没有就保留历史
    if new_entities.get("cities"):
        merged["cities"] = new_entities["cities"]
    else:
        # 确保 key 存在且是 list
        if not isinstance(merged.get("cities"), list):
            merged["cities"] = merged.get("cities") or []

    # duration_days：本轮明确给出就覆盖，否则保留历史
    if new_entities.get("duration_days") not in [None, "", 0]:
        merged["duration_days"] = new_entities["duration_days"]

    # dates：本轮有就覆盖（你也可以改成“本轮无则保留”，看产品策略）
    if new_entities.get("dates"):
        merged["dates"] = new_entities["dates"]

    # preferences：并集去重（保持顺序）
    old_prefs = merged.get("preferences") or []
    new_prefs = new_entities.get("preferences") or []
    if new_prefs:
        merged["preferences"] = list(dict.fromkeys(old_prefs + new_prefs))
    else:
        if not isinstance(merged.get("preferences"), list):
            merged["preferences"] = merged.get("preferences") or []

    # budget：本轮有新值就覆盖
    if new_entities.get("budget"):
        merged["budget"] = new_entities["budget"]

    # ✅ 累积排除的景点（跨轮继承）
    old_excluded = list(state.excluded_places or [])
    new_excluded = new_entities.get("excluded_places") or []

    # ✅ 累积想去的景点（跨轮继承）
    old_included = list(state.included_places or [])
    new_included = new_entities.get("included_places") or []
    merged_included = list(dict.fromkeys(old_included + new_included))  # 去重保序

    # ✅ 如果用户说"还是想去XX"，从排除列表中移除
    merged_excluded = []
    for place in dict.fromkeys(old_excluded + new_excluded):
        # 检查是否与任何 included_place 匹配（模糊匹配）
        should_exclude = True
        for inc in merged_included:
            if inc in place or place in inc:
                should_exclude = False
                break
        if should_exclude:
            merged_excluded.append(place)

    return {
        "intent": intent,
        "entities": merged,
        "excluded_places": merged_excluded,
        "included_places": merged_included,
        "current_node": "intent_recognition",
    }


def node_clarify_gate(state: GraphState) -> dict:
    """
    节点1.5: 澄清门控（B策略：尽量少问，先给方案）
    规则：
    - general_qa: 不强制要求城市，直接继续
    - trip_planning/clothing_advice 缺城市：必须问，直接结束本轮
    - 不缺城市：继续（clarify_only=False）
    """
    entities = state.entities or {}
    cities = entities.get("cities") or []
    intent_type = state.intent.intent_type if state.intent else "unknown"

    # ✅ general_qa 不需要强制城市，直接放行
    if intent_type == "general_qa":
        return {
            "clarify_only": False,
            "current_node": "clarify_gate",
        }

    # ✅ general_chat / unknown 也不需要城市
    if intent_type in ["general_chat", "unknown"]:
        return {
            "clarify_only": False,
            "current_node": "clarify_gate",
        }

    # 只有 trip_planning 和 clothing_advice 需要城市
    if not cities and intent_type in ["trip_planning", "clothing_advice"]:
        # 强制问城市（只问最关键的）
        q = "你想去哪个城市/目的地？（例如：武汉/成都/上海）"
        return {
            "final_response": f"为了给你更合适的方案，我还需要确认一个信息：\n- {q}",
            "clarify_only": True,
            "current_node": "clarify_gate",
        }

    # ✅ follow-up 纠偏：用户说"结合之前行程/出行计划/上面的计划" => 优先按穿搭继续
    # 典型：上一轮刚生成 trip_plan，用户问"按行程给配套穿搭/带什么衣服"
    text_raw = f"{state.user_input or ''}\n{state.rewritten_query or ''}"
    text = text_raw.lower()

    hit_plan = any(k in text for k in ["之前", "上面", "刚才", "行程", "出行计划", "旅行计划", "按行程", "结合行程"])
    hit_outfit = any(k in text for k in ["穿搭", "衣服", "带什么", "怎么穿", "配套", "outfit", "穿什么"])

    # 如果已有行程，并且用户在引用"行程/之前"，就把它当成穿搭 follow-up
    if state.trip_plan and hit_plan and (hit_outfit or "建议" in text):
        # 方式1：直接覆写 intent（最有效）
        if state.intent:
            state.intent.intent_type = "clothing_advice"
            state.intent.confidence = max(getattr(state.intent, "confidence", 0.0), 0.9)
        intent_type = "clothing_advice"

    # ✅ 新增：穿搭配套需要活动/场景
    text = f"{state.user_input or ''}\n{state.rewritten_query or ''}".lower()
    wants_activity_based_outfit = any(k in text for k in [
        "根据我的活动", "按活动", "按我的活动", "活动和气温", "场景", "配套", "穿搭方案", "一身", "outfit"
    ])

    prefs = entities.get("preferences") or []
    if intent_type == "clothing_advice" and wants_activity_based_outfit and not prefs:
        q = "你这次主要会有哪些活动/场景？（例如：城市暴走/拍照、徒步登山、看展、夜景、亲子、商务/通勤）"
        return {
            "final_response": f"我可以按活动+气温给你每天一套配套穿搭。开始前我想确认：\n- {q}",
            "clarify_only": True,
            "current_node": "clarify_gate",
        }

    return {
        "clarify_only": False,
        "current_node": "clarify_gate",
    }


def node_load_memory(state: GraphState) -> dict:
    """
    节点2: 加载用户记忆
    """
    query = state.rewritten_query or state.user_input
    relevant_memories = user_memory.search_relevant(query, k=3)
    profile = user_memory.get_user_profile()

    return {
        "user_profile": {"relevant_memories": relevant_memories, "profile": profile},
        "current_node": "load_memory",
    }


def node_fetch_weather(state: GraphState) -> dict:
    """
    节点3: 获取天气数据
    - 行程规划：实时天气
    - 穿搭建议：若用户问 N 天游，则拉取未来 N 天预报
    """
    entities = state.entities or {}
    cities = entities.get("cities", []) or []
    duration_days = int(entities.get("duration_days") or 1)

    intent_type = state.intent.intent_type if state.intent else "unknown"
    weather_data = {}

    for city in cities:
        if intent_type == "clothing_advice" and duration_days > 1:
            weather_data[city] = weather_tool.get_forecast(city, days=duration_days)
        else:
            weather_data[city] = weather_tool.get_weather(city)

    if not cities:
        if intent_type == "clothing_advice" and duration_days > 1:
            weather_data["北京"] = weather_tool.get_forecast("北京", days=duration_days)
        else:
            weather_data["北京"] = weather_tool.get_weather("北京")

    return {"weather_data": weather_data, "current_node": "fetch_weather"}


def node_clothing_advice(state: GraphState) -> dict:
    """
    节点4a: 多日穿搭建议（方案 B）
    """
    weather_data = state.weather_data or {}
    entities = state.entities or {}
    duration_days = int(entities.get("duration_days") or 1)

    # 取第一个城市
    if weather_data:
        city = list(weather_data.keys())[0]
        wobj = weather_data[city]
    else:
        city = "北京"
        wobj = weather_tool.get_weather(city)

    # 用户上下文（记忆 + 结构化偏好/活动）
    entities = state.entities or {}
    prefs = entities.get("preferences") or []

    user_context_parts = []

    # ✅ 把已生成的行程摘要喂给穿搭模型：让它按“当天活动强度/场景”配套
    if state.trip_plan and getattr(state.trip_plan, "days", None):
        lines = []
        for d in state.trip_plan.days:
            act_names = _extract_day_activity_names(d)
            if act_names:
                lines.append(f"- {d.date} {d.city}: " + "、".join(act_names[:4]))
        if lines:
            user_context_parts.append("已生成行程（用于配套穿搭）:\n" + "\n".join(lines))

    # ✅ 结构化：把“活动/场景/偏好”显式写出来
    if prefs:
        user_context_parts.append(f"活动/场景/偏好: {', '.join(prefs)}")

    # 保留用户原话（用于补充细节）
    if state.user_input:
        user_context_parts.append(f"用户原话: {state.user_input}")

    # 追加记忆
    if state.user_profile and state.user_profile.get("relevant_memories"):
        memories = [m["content"] for m in state.user_profile["relevant_memories"] if m.get("content")]
        if memories:
            user_context_parts.append(f"用户偏好/记忆: {', '.join(memories)}")

    user_context = "\n".join(user_context_parts).strip() or "日常出行"


    # 统一成 list[WeatherInfo]
    if isinstance(wobj, list):
        forecast = wobj[: max(1, duration_days)]
        header = f"🌤️ **{city}** 未来 {len(forecast)} 天穿搭建议"
    else:
        forecast = [wobj]
        header = f"🌤️ **{wobj.city}** 今日穿搭建议"

    blocks = [header]

    for i, w in enumerate(forecast, 1):
        raw = w.raw_data or {}
        date_text = (raw.get("date") or "").strip()
        # 过滤 mock 的占位日期（Day1/Day2 这类）
        if date_text.lower() in {f"day{i}".lower(), f"day {i}".lower()}:
            date_text = ""
        daytemp = raw.get("daytemp")
        nighttemp = raw.get("nighttemp")
        dayweather = raw.get("dayweather") or w.weather
        nightweather = raw.get("nightweather")

        title = f"### Day {i}" + (f" - {date_text}" if date_text else "")
        if daytemp is not None and nighttemp is not None:
            line = f"- 预报：{dayweather}（白天 {daytemp}°C / 夜间 {nighttemp}°C）"
            if nightweather and nightweather != dayweather:
                line += f"，夜间：{nightweather}"
        else:
            line = f"- 预报：{w.weather}（体感参考约 {w.temperature}°C）"

        # ✅ 逐日行程绑定：取第 i 天的活动，单独喂给模型
        activity_names = []
        if state.trip_plan and getattr(state.trip_plan, "days", None) and len(state.trip_plan.days) >= i:
            d = state.trip_plan.days[i - 1]
            activity_names = _extract_day_activity_names(d)

        tags = _infer_activity_tags(activity_names)

        if activity_names:
            day_plan_text = "当天活动: " + "、".join(activity_names[:6])
        else:
            day_plan_text = "当天活动: 未提供详细行程（按城市步行/通用旅行建议）"

        day_user_context = (
                user_context
                + "\n\n"
                + f"【第{i}天行程绑定】\n{day_plan_text}\n活动标签: {', '.join(tags)}\n"
                + "硬性要求：summary 必须以“考虑到你今天要【活动/地点】（【活动标签】），结合当日【白天xx°C/夜间xx°C+天气】，建议：”开头。"
        )

        advice = clothing_advisor.advise(w, day_user_context)
        formatted = clothing_advisor.format_advice(advice)

        blocks.append("\n".join([title, line, "", formatted]).strip())

    response = "\n\n".join(blocks).strip()

    return {
        "clothing_advice": response,
        "final_response": response,
        "current_node": "clothing_advice",
    }


def _normalize_activity(a: Any) -> dict:
    """
    把任意 activity 结构归一化成统一 dict：
    {
      "time": "09:00",
      "name": "中山陵",
      "desc": "...",
      "meta": {...}
    }
    最少保证 name 存在
    """
    # activity 是字符串
    if isinstance(a, str):
        name = a.strip() or "行程活动"
        return {"time": "", "name": name, "description": "", "meta": {"raw": a}}

    # activity 是 dict（最常见）
    if isinstance(a, dict):
        # 兼容不同 key
        name = ""
        for k in ("name", "title", "place", "spot", "activity"):
            v = a.get(k)
            if isinstance(v, str) and v.strip():
                name = v.strip()
                break

        time = a.get("time") if isinstance(a.get("time"), str) else ""

        desc = ""
        for k in ("desc", "description", "note", "tips"):
            v = a.get(k)
            if isinstance(v, str) and v.strip():
                desc = v.strip()
                break

        meta = {
            k: v
            for k, v in a.items()
            if k
            not in (
                "time",
                "name",
                "title",
                "place",
                "spot",
                "activity",
                "desc",
                "description",
                "note",
                "tips",
            )
        }

        return {"time": time, "name": name or "行程活动", "description": desc, "meta": meta}

    # 其它类型兜底
    s = str(a).strip()
    return {"time": "", "name": s or "行程活动", "description": "", "meta": {"raw": s}}


def _normalize_activities(activities: Any) -> list[dict]:
    if activities is None:
        return []
    if isinstance(activities, list):
        return [_normalize_activity(x) for x in activities]
    return [_normalize_activity(activities)]

def node_trip_planning(state: GraphState) -> dict:
    """
    节点4b: 行程规划 (trip_planning 分支)
    """
    from agents.planner import trip_planner

    weather_payload = {}
    for city, w in (state.weather_data or {}).items():
        if isinstance(w, list):
            weather_payload[city] = [x.model_dump() for x in w]
        elif w is None:
            weather_payload[city] = None
        else:
            weather_payload[city] = w.model_dump()

    # ✅ 检测用户的调整类型
    user_input_lower = (state.user_input or "").lower()
    adjustment_hint = ""
    if state.trip_plan:  # 如果已有行程，检测用户想要什么调整
        if any(kw in user_input_lower for kw in ["预算高", "预算充足", "贵点", "更好", "高端", "豪华"]):
            adjustment_hint = "用户预算提高了，请推荐更高端、更有特色的景点和餐厅，必须与之前的行程有明显区别！"
        elif any(kw in user_input_lower for kw in ["预算低", "预算不够", "便宜", "省钱", "经济", "实惠"]):
            adjustment_hint = "用户预算降低了，请推荐更经济实惠的景点和餐厅，优先免费景点和平价美食，必须与之前的行程有明显区别！"
        elif any(kw in user_input_lower for kw in ["热闹", "人多", "繁华", "商业"]):
            adjustment_hint = "用户想要更热闹的地方，请推荐人气旺、商业繁华的景点，必须与之前的行程有明显区别！"
        elif any(kw in user_input_lower for kw in ["安静", "清净", "人少", "小众"]):
            adjustment_hint = "用户想要更安静的地方，请推荐人少、小众的景点，必须与之前的行程有明显区别！"
        elif any(kw in user_input_lower for kw in ["好玩", "有趣", "刺激", "特色"]):
            adjustment_hint = "用户想要更好玩的地方，请推荐更有特色、更有趣的景点和活动，必须与之前的行程有明显区别！"

    context = {
        "user_input": state.user_input,
        "entities": state.entities,
        "weather_data": weather_payload,
        "user_profile": state.user_profile,
        "previous_plan": state.trip_plan,  # 传递之前的行程，用于修改/调整
        "excluded_places": state.excluded_places,  # 传递排除的景点列表
        "included_places": state.included_places,  # 传递想去的景点列表
        "adjustment_hint": adjustment_hint,  # 传递调整提示
    }

    plan = trip_planner.plan(context)

    # ✅ 新增：统一 activities 字段（保证每个 dict 至少有 name）
    if plan and getattr(plan, "days", None):
        for day in plan.days:
            day.activities = _normalize_activities(getattr(day, "activities", None))

    needs_replan = False
    for day in plan.days:
        if day.weather and "雨" in day.weather.weather:
            day.risk_level = "medium"
            needs_replan = True
        if day.weather and ("暴" in day.weather.weather or "雪" in day.weather.weather):
            day.risk_level = "high"
            needs_replan = True

    return {
        "trip_plan": plan,
        "needs_replan": needs_replan,
        "current_node": "trip_planning",
    }


def node_risk_assessment(state: GraphState) -> dict:
    """
    节点5: 风险评估与备选方案
    """
    if not state.needs_replan or not state.trip_plan:
        return {"current_node": "risk_assessment"}

    from agents.planner import trip_planner

    plan = state.trip_plan
    for day in plan.days:
        if day.risk_level in ["medium", "high"]:
            day.backup_plan = trip_planner.generate_backup(day)

    return {"trip_plan": plan, "current_node": "risk_assessment"}


def node_format_response(state: GraphState) -> dict:
    """
    节点6: 格式化最终响应
    """
    intent_type = state.intent.intent_type if state.intent else "unknown"

    if state.final_response:
        return {"current_node": "format_response"}

    if intent_type == "clothing_advice":
        return {"current_node": "format_response"}

    elif intent_type == "trip_planning" and state.trip_plan:
        response = format_trip_plan(state.trip_plan)

        # 过滤掉已经有答案的澄清问题
        questions = state.clarifying_questions or []
        entities = state.entities or {}
        filtered_questions = []
        for q in questions:
            q_lower = q.lower()
            # 如果已有城市，跳过城市相关问题
            if entities.get("cities") and any(kw in q_lower for kw in ["城市", "目的地", "去哪", "区域"]):
                continue
            # 如果已有天数，跳过天数相关问题
            if entities.get("duration_days") and any(kw in q_lower for kw in ["几天", "天数", "多久"]):
                continue
            filtered_questions.append(q)

        if filtered_questions and not state.clarify_only:
            response += "\n\n---\n### 需要你确认\n" + "\n".join([f"- {q}" for q in filtered_questions[:2]])

        return {"final_response": response, "current_node": "format_response"}

    else:
        return {
            "final_response": "抱歉，我不太理解您的需求。我可以帮您：\n1. 📌 规划多日行程\n2. 👔 根据天气给出穿搭建议\n\n请告诉我您想做什么？",
            "current_node": "format_response",
        }


def node_update_memory(state: GraphState) -> dict:
    """
    节点7: 更新用户记忆 (后台任务)
    """
    conversation = f"用户: {state.user_input}\n助手: {state.final_response}"
    extracted = entity_extractor.extract(conversation)

    rs = state.rewrite_slots or {}
    budget_text = rs.get("budget_text")

    if budget_text:
        user_memory.add_preference(
            content=str(budget_text),
            category="budget",
            source="rewrite",
        )

    def _get(pref, key: str, default=None):
        # dict
        if isinstance(pref, dict):
            return pref.get(key, default)
        # pydantic v2
        if hasattr(pref, "model_dump"):
            return pref.model_dump().get(key, default)
        # pydantic v1
        if hasattr(pref, "dict"):
            return pref.dict().get(key, default)
        # 普通对象 / dataclass
        return getattr(pref, key, default)

    if getattr(extracted, "has_new_info", False):
        for pref in (getattr(extracted, "preferences", None) or []):
            content = _get(pref, "content")
            category = _get(pref, "category")
            if not content or not category:
                continue

            user_memory.add_preference(
                content=str(content),
                category=str(category),
                source="conversation",
            )

    user_memory.add_memory(
        f"用户询问: {state.user_input[:100]}...",
        memory_type="interaction",
    )

    return {"current_node": "update_memory"}


def node_general_qa(state: GraphState) -> dict:
    """
    节点: 通用旅行问答（general_qa）
    处理景点介绍、美食推荐、交通问题、健康出行建议等
    """
    from langchain_core.prompts import ChatPromptTemplate
    from utils.llm import get_llm

    user_input = state.user_input or ""
    entities = state.entities or {}

    # 判断是否涉及健康问题
    has_health_concern = entities.get("has_health_concern", False)
    query_subject = entities.get("query_subject", "")

    # 构建上下文信息
    context_parts = []

    # 智能判断：如果之前有行程，且问题可能与行程相关，则提供上下文
    if state.trip_plan and getattr(state.trip_plan, "days", None):
        # 提取行程中的景点和活动
        trip_places = []
        trip_foods = []
        for day in state.trip_plan.days:
            if hasattr(day, "activities") and day.activities:
                for act in day.activities:
                    name = act.get("name", "") if isinstance(act, dict) else str(act)
                    if name:
                        trip_places.append(name)

        if trip_places:
            context_parts.append(f"用户之前的行程包含这些地点/活动: {', '.join(trip_places[:10])}")

    # 如果有城市信息
    cities = entities.get("cities", [])
    if cities:
        context_parts.append(f"相关城市: {', '.join(cities)}")

    # 用户画像/偏好
    if state.user_profile and state.user_profile.get("relevant_memories"):
        memories = [m["content"] for m in state.user_profile["relevant_memories"] if m.get("content")]
        if memories:
            context_parts.append(f"用户偏好: {', '.join(memories[:5])}")

    context_str = "\n".join(context_parts) if context_parts else "无额外上下文"

    # 构建 prompt
    system_prompt = """你是 IntelliGo 智能出行助手，专门回答旅行相关的问题。

## 你的职责
- 回答景点/地点的详细介绍（历史、特色、游玩建议）
- 回答美食相关问题（特色菜、口味、推荐餐厅）
- 回答交通相关问题（怎么去、多久、交通方式选择）
- 回答健康出行问题（特定人群是否适合某活动/食物）
- 回答旅行常识（证件、安全、注意事项）

## 回答要求
- 回答要简洁实用，重点突出
- 如果有上下文信息，可以结合用户之前的行程来回答
- 语气亲切自然，像朋友推荐一样

## 上下文信息
{context}

{health_disclaimer}"""

    health_disclaimer = ""
    if has_health_concern:
        health_disclaimer = """## 健康问题特别提醒
涉及健康相关问题时，你需要：
1. 给出通用的建议和参考信息
2. 在回答末尾添加免责声明：「以上建议仅供参考，具体请咨询专业医生，根据个人身体状况做出判断。」"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{user_input}")
    ])

    llm = get_llm(temperature=0.7)
    chain = prompt | llm

    response = chain.invoke({
        "context": context_str,
        "health_disclaimer": health_disclaimer,
        "user_input": user_input
    })

    # 提取回答内容
    answer = response.content if hasattr(response, "content") else str(response)

    return {
        "final_response": answer,
        "current_node": "general_qa",
    }


# ====== 你原本的 format_trip_plan 如果在其他文件，这里保持引用即可 ======
def format_trip_plan(plan: TripPlan) -> str:
    """
    简单的行程格式化（如果你项目里已有同名函数，可删除此处）
    """
    lines = []
    title = plan.title or "行程规划"
    lines.append(f"## {title}")

    for i, day in enumerate(plan.days, 1):
        header = f"### Day {i} - {day.city} {day.date}".strip()
        lines.append(header)

        if day.weather:
            lines.append(f"- 天气: {day.weather.weather} {day.weather.temperature}°C")
        lines.append(f"- 风险等级: {day.risk_level}")

        if day.backup_plan:
            lines.append(f"- 备选方案: {day.backup_plan}")

        if day.activities:
            lines.append("")
            for act in day.activities:
                t = act.get("time", "")
                name = act.get("name", "")
                desc = act.get("description", "")
                lines.append(f"- {t} {name} {desc}".strip())

        lines.append("")

    if plan.tips:
        lines.append("---")
        lines.append("### 小贴士")
        for tip in plan.tips:
            lines.append(f"- {tip}")

    if plan.total_budget_estimate:
        lines.append(f"\n**预估总花费:** {plan.total_budget_estimate}")

    return "\n".join(lines).strip()
