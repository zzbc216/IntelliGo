from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core.assistant import run_one_turn, validate_config_or_raise
from core.session_store import InMemorySessionStore
from config import config
from memory.vector_store import UserMemory


app = FastAPI(title="IntelliGo API", version="0.1.0")

# 会话存储（个人使用先内存即可）
store = InMemorySessionStore(ttl_seconds=60 * 60 * 24 * 7)


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    # 额外信息（可用于前端调试/展示）
    current_node: str = ""
    need_clarification: bool = False
    clarifying_questions: list[str] = Field(default_factory=list)
    intent_type: str = ""
    entities: dict = Field(default_factory=dict)


@app.get("/", response_class=HTMLResponse)
def index():
    """
    直接返回移动端网页
    """
    web_path = os.path.join(os.path.dirname(__file__), "web", "index.html")
    with open(web_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/health")
def health():
    """
    健康检查：也验证配置是否齐全
    """
    try:
        validate_config_or_raise()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        validate_config_or_raise()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # ===== 管理员指令拦截：清空全部（Chroma + 当前会话）=====
    text = (req.message or "").strip()
    if text.startswith("/purge_all"):
        # 格式：/purge_all <token>
        parts = text.split(maxsplit=1)
        token = parts[1].strip() if len(parts) == 2 else ""

        if not config.purge_token:
            raise HTTPException(status_code=500, detail="PURGE_TOKEN 未设置，禁止清空操作。")

        if token != config.purge_token:
            return ChatResponse(
                session_id=req.session_id,
                reply="口令错误：未执行清空。",
                current_node="admin_command",
            )

        # 1) 清空 Chroma 持久化数据
        UserMemory.clear_all_persisted_data()
        # 2) 重置 graph/nodes.py 中的全局 user_memory 实例
        from graph import nodes
        nodes.user_memory.reset()
        # 3) 清空当前会话 state（避免还显示旧画像/旧状态）
        store.reset(req.session_id)

        return ChatResponse(
            session_id=req.session_id,
            reply="已清空全部用户画像/记忆（Chroma 数据目录已清空），并重置当前会话。",
            current_node="admin_command",
        )
    # ===== 拦截结束 =====

    last_state = store.get(req.session_id)
    final_state, reply = run_one_turn(req.message, last_state=last_state)
    store.set(req.session_id, final_state)

    intent_type = ""
    intent = final_state.get("intent")
    if isinstance(intent, dict):
        intent_type = intent.get("intent_type") or ""
    else:
        # 有些情况下可能是 Pydantic/BaseModel 或其他对象；尽量兼容
        try:
            intent_type = getattr(intent, "intent_type", "") or ""
        except Exception:
            intent_type = ""

    return ChatResponse(
        session_id=req.session_id,
        reply=reply,
        current_node=final_state.get("current_node") or "",
        need_clarification=bool(final_state.get("need_clarification")),
        clarifying_questions=final_state.get("clarifying_questions") or [],
        intent_type=intent_type,
        entities=final_state.get("entities") or {},
    )


@app.get("/api/profile")
def profile(session_id: str):
    """
    获取用户画像（格式化版本，适合前端展示）

    返回结构：
    {
        "session_id": "xxx",
        "formatted": {
            "summary": "偏安静、喜欢博物馆、避免爬山、预算重性价比",
            "cards": [
                {
                    "id": "travel_habits",
                    "title": "出行习惯",
                    "icon": "🚶",
                    "items": [
                        {"text": "安静、不拥挤", "source": {...}}
                    ]
                },
                ...
            ],
            "raw_count": 10
        },
        "user_profile": {...}  # 原始数据（兼容旧版）
    }
    """
    st = store.get(session_id) or {}

    # 获取格式化画像（直接从 Chroma 读取，不依赖 session state）
    user_memory = UserMemory()
    formatted = user_memory.get_formatted_profile()

    return {
        "session_id": session_id,
        "formatted": formatted,
        "user_profile": st.get("user_profile") or {}  # 保留原始数据兼容旧版
    }


@app.post("/api/reset")
def reset(session_id: str):
    store.reset(session_id)
    return {"ok": True, "session_id": session_id}


@app.post("/api/clear_profile")
def clear_profile():
    """
    清空用户画像（一键清空，无需口令）
    """
    # 1) 清空 Chroma 持久化数据
    UserMemory.clear_all_persisted_data()
    # 2) 重置 graph/nodes.py 中的全局 user_memory 实例
    from graph import nodes
    nodes.user_memory.reset()

    return {"ok": True, "message": "用户画像已清空"}


@app.get("/api/suggestions")
def suggestions(session_id: str):
    """
    根据用户画像生成个性化问题建议
    - 画像不为空：AI 生成 3 个个性化问题
    - 画像为空：返回空数组，前端使用 fallback
    """
    user_memory = UserMemory()
    formatted = user_memory.get_formatted_profile()

    # 如果画像为空，返回空数组让前端走 fallback
    if not formatted.get("cards") or len(formatted.get("cards", [])) == 0:
        return {
            "session_id": session_id,
            "questions": [],
            "title": ""
        }

    # 画像不为空，用 LLM 生成个性化问题
    from utils.llm import get_llm
    from langchain_core.prompts import ChatPromptTemplate
    from pydantic import BaseModel
    import json

    # 构建画像摘要
    profile_parts = []
    for card in formatted.get("cards", []):
        items_text = [item["text"] for item in card.get("items", [])[:3]]
        if items_text:
            profile_parts.append(f"{card['title']}: {', '.join(items_text)}")

    profile_summary = "\n".join(profile_parts)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是 IntelliGo 智能出行助手。根据用户的偏好画像，生成 3 个个性化的问题建议。

## 要求
1. 问题要**贴合用户偏好**，让用户感到"懂我"
2. 3 个问题分别覆盖：**行程规划**、**出行/交通**、**穿搭建议**
3. 问题要具体、自然，像用户会说的话
4. 每个问题不超过 25 个字

## 输出格式
直接输出 JSON 数组，不要有其他内容：
["问题1", "问题2", "问题3"]

## 示例
用户偏好：喜欢安静、博物馆、预算敏感
输出：["周末想找个安静的地方逛逛，有推荐吗？", "去博物馆一天怎么安排最省心？", "明天降温了，帮我搭一套休闲穿搭"]"""),
        ("human", """用户偏好画像：
{profile}

请生成 3 个个性化问题建议：""")
    ])

    try:
        llm = get_llm(temperature=0.8)
        chain = prompt | llm
        response = chain.invoke({"profile": profile_summary})

        # 解析 JSON
        content = response.content if hasattr(response, "content") else str(response)
        # 清理可能的 markdown 代码块
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
        if content.endswith("```"):
            content = content.rsplit("```", 1)[0]
        content = content.strip()

        questions = json.loads(content)

        if isinstance(questions, list) and len(questions) >= 3:
            return {
                "session_id": session_id,
                "questions": questions[:3],
                "title": "💡 根据你的偏好，试试这些问法："
            }
    except Exception as e:
        # 解析失败，返回空让前端走 fallback
        pass

    return {
        "session_id": session_id,
        "questions": [],
        "title": ""
    }


# 可选：挂载静态目录（如果你后面要加 css/js 文件）
# 当前 index.html 内联了，不依赖这个也能跑
static_dir = os.path.join(os.path.dirname(__file__), "web", "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

