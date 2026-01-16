"""
IntelliGo 配置管理
"""
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@dataclass
class Config:
    """全局配置"""
    # OpenAI 兼容 API
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL"))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    # 高德 API
    amap_api_key: str = field(default_factory=lambda: os.getenv("AMAP_API_KEY", ""))

    # Chroma
    chroma_persist_dir: str = field(default_factory=lambda: os.path.join(BASE_DIR, "data", "chroma_db"))
    chroma_collection_name: str = "user_preferences"

    # 管理指令口令
    purge_token: str = field(default_factory=lambda: os.getenv("PURGE_TOKEN", ""))

    # 系统设置
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "true").lower() == "true")

    def validate(self) -> list[str]:
        """验证必要配置"""
        errors = []
        if not self.openai_api_key:
            errors.append("❌ OPENAI_API_KEY 未设置")
        if not self.amap_api_key:
            errors.append("⚠️ AMAP_API_KEY 未设置 (天气/地图功能将使用 Mock 数据)")
        return errors

    def __post_init__(self):
        """初始化后打印配置（调试用）"""
        if self.debug:
            print(f"🔧 Config loaded:")
            print(f"   - API Base: {self.openai_base_url}")
            print(f"   - Model: {self.openai_model}")


# 全局单例
config = Config()
