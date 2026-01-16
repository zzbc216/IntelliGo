from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class SessionRecord:
    state: Optional[Dict[str, Any]]
    updated_at: float


class InMemorySessionStore:
    """
    最简会话存储：适合单进程部署/个人使用
    注意：进程重启会丢会话；多进程会话不共享（上线稳定版建议换 Redis）
    """

    def __init__(self, ttl_seconds: int = 60 * 60 * 24):
        self._db: Dict[str, SessionRecord] = {}
        self._ttl = ttl_seconds

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        rec = self._db.get(session_id)
        if not rec:
            return None

        if time.time() - rec.updated_at > self._ttl:
            # 过期清理
            self._db.pop(session_id, None)
            return None

        return rec.state

    def set(self, session_id: str, state: Optional[Dict[str, Any]]) -> None:
        self._db[session_id] = SessionRecord(state=state, updated_at=time.time())

    def reset(self, session_id: str) -> None:
        self._db.pop(session_id, None)
