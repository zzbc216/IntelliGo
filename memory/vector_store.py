"""
基于 Chroma 的用户记忆系统
"""
import os
from datetime import datetime
from difflib import SequenceMatcher
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config import config
from utils.llm import get_embeddings  # ⭐ 使用统一的 embedding 获取函数
import shutil


def _text_similarity(a: str, b: str) -> float:
    """计算两段文本的相似度 (0~1)"""
    return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()


class UserMemory:
    """用户记忆管理器"""

    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id

        # 确保目录存在
        os.makedirs(config.chroma_persist_dir, exist_ok=True)

        # ⭐ 使用统一封装的 embeddings
        self.embeddings = get_embeddings()

        # 初始化 Chroma
        self.vectorstore = Chroma(
            collection_name=f"{config.chroma_collection_name}_{user_id}",
            embedding_function=self.embeddings,
            persist_directory=config.chroma_persist_dir
        )

    def add_preference(self, content: str, category: str, source: str = "conversation",
                       similarity_threshold: float = 0.75):
        """
        添加用户偏好（自动去重：已存在相似内容则跳过）

        Args:
            content: 偏好内容，如 "喜欢安静的咖啡厅"
            category: 类别，如 "dining", "activity", "travel_style"
            source: 来源
            similarity_threshold: 文本相似度阈值，超过则视为重复不添加
        """
        content = content.strip()
        if not content:
            return

        # 检查是否已存在相似内容
        existing = self.vectorstore.get(where={"type": "preference"})
        if existing and existing.get("documents"):
            for doc_content in existing["documents"]:
                if _text_similarity(content, doc_content) >= similarity_threshold:
                    return  # 已存在相似内容，跳过

        doc = Document(
            page_content=content,
            metadata={
                "user_id": self.user_id,
                "category": category,
                "source": source,
                "timestamp": datetime.now().isoformat(),
                "type": "preference"
            }
        )
        self.vectorstore.add_documents([doc])

    def add_memory(self, content: str, memory_type: str = "interaction"):
        """添加交互记忆"""
        doc = Document(
            page_content=content,
            metadata={
                "user_id": self.user_id,
                "type": memory_type,
                "timestamp": datetime.now().isoformat()
            }
        )
        self.vectorstore.add_documents([doc])

    def search_relevant(self, query: str, k: int = 5,
                         min_score: float = 0.3,
                         dedup_threshold: float = 0.7) -> list[dict]:
        """
        搜索相关记忆（带相似度门槛和去重）

        Args:
            query: 查询文本
            k: 最多返回条数
            min_score: 最低相似度门槛，低于此值的结果不返回
            dedup_threshold: 去重阈值，与已选内容相似度超过此值则跳过

        Returns:
            [{"content": "...", "category": "...", "score": 0.85}, ...]
        """
        # 多取一些候选，后续去重可能会过滤掉部分
        results = self.vectorstore.similarity_search_with_score(query, k=k * 2)

        selected = []
        selected_contents = []

        for doc, score in results:
            sim_score = round(1 - score, 3)  # 转换为相似度

            # 门槛1：相似度太低，跳过
            if sim_score < min_score:
                continue

            content = doc.page_content

            # 门槛2：与已选内容重复，跳过
            is_duplicate = False
            for existing in selected_contents:
                if _text_similarity(content, existing) >= dedup_threshold:
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            selected.append({
                "content": content,
                "category": doc.metadata.get("category", "unknown"),
                "type": doc.metadata.get("type", "unknown"),
                "score": sim_score
            })
            selected_contents.append(content)

            # 达到目标数量后停止
            if len(selected) >= k:
                break

        return selected

    def get_user_profile(self, dedup_threshold: float = 0.7) -> dict:
        """
        获取用户画像摘要（自动去重）

        Args:
            dedup_threshold: 去重阈值，相似度超过此值的只保留一条
        """
        # 获取所有偏好类记忆
        all_docs = self.vectorstore.get(
            where={"type": "preference"}
        )

        profile = {
            "preferences": [],
            "categories": {}
        }

        if all_docs and all_docs.get("documents"):
            seen_contents = []  # 用于去重

            for i, content in enumerate(all_docs["documents"]):
                # 检查是否与已添加内容重复
                is_duplicate = False
                for seen in seen_contents:
                    if _text_similarity(content, seen) >= dedup_threshold:
                        is_duplicate = True
                        break

                if is_duplicate:
                    continue

                seen_contents.append(content)
                metadata = all_docs["metadatas"][i] if all_docs.get("metadatas") else {}
                category = metadata.get("category", "general")

                profile["preferences"].append(content)

                if category not in profile["categories"]:
                    profile["categories"][category] = []
                profile["categories"][category].append(content)

        return profile

    def get_formatted_profile(self, dedup_threshold: float = 0.7) -> dict:
        """
        获取格式化的用户画像（适合前端展示）

        返回结构：
        {
            "summary": "一句话总结",
            "cards": [
                {
                    "id": "travel_habits",
                    "title": "出行习惯",
                    "icon": "🚶",
                    "items": [
                        {"text": "喜欢安静", "source": {...}}
                    ]
                },
                ...
            ],
            "raw_count": 10  # 原始记录总数
        }
        """
        # 定义卡片类别映射
        CARD_CONFIG = [
            {
                "id": "travel_habits",
                "title": "出行习惯",
                "icon": "🚶",
                "categories": ["travel_style", "activity", "style"],
                "keywords": ["习惯", "喜欢", "偏好", "节奏", "方式"]
            },
            {
                "id": "favorite_places",
                "title": "喜欢去的地方",
                "icon": "📍",
                "categories": ["place", "destination", "location", "spot"],
                "keywords": ["景点", "地方", "去", "博物馆", "美术馆", "咖啡", "公园"]
            },
            {
                "id": "dislikes",
                "title": "不喜欢/需要避免",
                "icon": "⚠️",
                "categories": ["dislike", "avoid", "restriction"],
                "keywords": ["不喜欢", "不想", "避免", "不要", "讨厌", "膝盖", "不适"]
            },
            {
                "id": "budget",
                "title": "预算与消费",
                "icon": "💰",
                "categories": ["budget", "price", "cost"],
                "keywords": ["预算", "性价比", "便宜", "贵", "消费", "花费", "元", "块"]
            },
            {
                "id": "food",
                "title": "饮食偏好",
                "icon": "🍜",
                "categories": ["dining", "food", "cuisine", "restaurant"],
                "keywords": ["吃", "餐", "美食", "口味", "辣", "甜", "素", "海鲜"]
            },
            {
                "id": "accommodation",
                "title": "住宿/交通偏好",
                "icon": "🏨",
                "categories": ["accommodation", "hotel", "transport", "traffic"],
                "keywords": ["住", "酒店", "民宿", "交通", "高铁", "飞机", "自驾"]
            }
        ]

        # 获取所有偏好记录（带元数据）
        all_docs = self.vectorstore.get(where={"type": "preference"})

        raw_records = []
        if all_docs and all_docs.get("documents"):
            for i, content in enumerate(all_docs["documents"]):
                metadata = all_docs["metadatas"][i] if all_docs.get("metadatas") else {}
                raw_records.append({
                    "content": content,
                    "category": metadata.get("category", "general"),
                    "timestamp": metadata.get("timestamp", ""),
                    "source": metadata.get("source", "unknown")
                })

        # 去重
        seen_contents = []
        unique_records = []
        for rec in raw_records:
            is_dup = False
            for seen in seen_contents:
                if _text_similarity(rec["content"], seen) >= dedup_threshold:
                    is_dup = True
                    break
            if not is_dup:
                seen_contents.append(rec["content"])
                unique_records.append(rec)

        # 分配到卡片
        cards = []
        used_indices = set()

        for card_cfg in CARD_CONFIG:
            items = []
            for idx, rec in enumerate(unique_records):
                if idx in used_indices:
                    continue

                content_lower = rec["content"].lower()
                category_lower = rec["category"].lower()

                # 匹配规则：类别匹配 或 关键词匹配
                matched = False

                # 类别匹配
                for cat in card_cfg["categories"]:
                    if cat in category_lower:
                        matched = True
                        break

                # 关键词匹配
                if not matched:
                    for kw in card_cfg["keywords"]:
                        if kw in content_lower:
                            matched = True
                            break

                if matched:
                    items.append({
                        "text": self._shorten_preference(rec["content"]),
                        "source": {
                            "original": rec["content"],
                            "category": rec["category"],
                            "timestamp": rec["timestamp"],
                            "source_type": rec["source"]
                        }
                    })
                    used_indices.add(idx)

            # 只添加有内容的卡片
            if items:
                cards.append({
                    "id": card_cfg["id"],
                    "title": card_cfg["title"],
                    "icon": card_cfg["icon"],
                    "items": items
                })

        # 生成一句话总结
        summary = self._generate_summary(cards)

        return {
            "summary": summary,
            "cards": cards,
            "raw_count": len(raw_records)
        }

    def _shorten_preference(self, text: str, max_len: int = 30) -> str:
        """将偏好文本缩短为简洁形式"""
        text = text.strip()

        # 移除常见前缀
        prefixes = ["用户", "我", "本人", "偏好", "喜欢", "希望"]
        for p in prefixes:
            if text.startswith(p):
                text = text[len(p):].lstrip("：:,，")

        # 截断
        if len(text) > max_len:
            text = text[:max_len] + "..."

        return text

    def _generate_summary(self, cards: list) -> str:
        """根据卡片内容生成一句话总结"""
        parts = []

        for card in cards:
            if not card["items"]:
                continue

            card_id = card["id"]
            items_text = [item["text"] for item in card["items"][:2]]  # 最多取2条

            if card_id == "travel_habits":
                parts.append("偏" + "、".join(items_text[:1]))
            elif card_id == "favorite_places":
                parts.append("喜欢" + "/".join(items_text[:1]))
            elif card_id == "dislikes":
                parts.append("避免" + items_text[0] if items_text else "")
            elif card_id == "budget":
                parts.append("预算" + items_text[0] if items_text else "")

        if not parts:
            return ""

        return "、".join(p for p in parts if p)

    def reset(self):
        """
        重置当前实例：删除 collection 中所有数据并重新初始化 vectorstore
        """
        # 删除当前 collection 的所有数据
        try:
            # 获取所有文档 ID 并删除
            all_data = self.vectorstore.get()
            if all_data and all_data.get("ids"):
                self.vectorstore.delete(ids=all_data["ids"])
        except Exception:
            pass

        # 重新初始化 vectorstore
        self.vectorstore = Chroma(
            collection_name=f"{config.chroma_collection_name}_{self.user_id}",
            embedding_function=self.embeddings,
            persist_directory=config.chroma_persist_dir
        )

    @staticmethod
    def clear_all_persisted_data():
        """
        清空整个 Chroma 持久化目录（会删除所有 collection/所有用户数据）
        """
        from config import config  # 避免循环 import 的话也可以放文件顶部

        if os.path.isdir(config.chroma_persist_dir):
            shutil.rmtree(config.chroma_persist_dir, ignore_errors=True)
        os.makedirs(config.chroma_persist_dir, exist_ok=True)


# ---- 测试 ----
if __name__ == "__main__":
    memory = UserMemory("test_user")

    # 添加一些测试偏好
    memory.add_preference("喜欢安静的咖啡厅", "dining")
    memory.add_preference("不喜欢爬山，膝盖不好", "activity")
    memory.add_preference("偏好文艺小众景点", "travel_style")
    memory.add_preference("预算敏感，喜欢性价比高的选择", "budget")

    # 搜索测试
    print("🔍 搜索 '周末去哪玩' 相关记忆:")
    results = memory.search_relevant("周末去哪玩，想找个安静的地方")
    for r in results:
        print(f"  - [{r['category']}] {r['content']} (相关度: {r['score']})")

    # 获取画像
    print("\n👤 用户画像:")
    profile = memory.get_user_profile()
    for cat, prefs in profile["categories"].items():
        print(f"  {cat}: {prefs}")
