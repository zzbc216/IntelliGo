"""
LLM 客户端统一封装
支持第三方 OpenAI 兼容 API
"""
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from config import config
from langchain_huggingface import HuggingFaceEmbeddings

def get_llm(temperature: float = 0.7, model: str | None = None,max_retries: int = 3,) -> ChatOpenAI:
    """
    获取 LLM 实例
    自动使用配置中的 base_url，支持第三方 API 代理
    """
    return ChatOpenAI(
        api_key=config.openai_api_key,
        base_url="https://www.dmxapi.cn/v1",
        model=model or config.openai_model,
        max_retries=max_retries,
        request_timeout=60,
        temperature=temperature,
    )


def get_structured_llm(output_schema, temperature: float = 0.0):
    """获取结构化输出的 LLM"""
    llm = get_llm(temperature=temperature)
    return llm.with_structured_output(output_schema)


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    获取 Embedding 模型

    注意：某些第三方 API 可能不支持 embedding，需要单独配置
    """
    return HuggingFaceEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    model_kwargs={
        "device": 'cpu',},
    encode_kwargs={
        'normalize_embeddings': True,  # 可选：归一化向量，有助于相似度计算
        "batch_size": 32
    }
)
