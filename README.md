# IntelliGo -- 智能出行规划助手

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/LangGraph-0.1+-green.svg" alt="LangGraph">
  <img src="https://img.shields.io/badge/FastAPI-0.100+-orange.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/ChromaDB-0.4+-purple.svg" alt="ChromaDB">
</div>

一款基于 LangGraph 构建的智能出行规划工具，专注提供个性化出行解决方案，支持多城市/多日行程规划、天气穿搭建议、用户偏好记忆、恶劣天气预警，解决传统出行规划信息零散、无记忆的痛点。

---

## ✨核心功能
- 📍个性化行程规划：多城市/多日定制、自定义预算，贴合安静/性价比等出行偏好
- 👔天气联动穿搭：根据目标城市实时天气，生成穿搭建议
- 🧠持久化偏好记忆：跨会话保存用户出行习惯
- ⚠️主动风险预警：恶劣天气提醒并生成备选行程方案
- 📊会话调试：内置指令查看用户画像与会话状态，便捷管理

---

## 🚀快速开始

### 环境要求
- Python 3.10 及以上版本

### 1) 克隆仓库
```
git clone <https://github.com/zzbc216/IntelliGo.git>
cd IntelliGo
```

### 2) 安装依赖
```bash
pip install -r requirements.txt
```

### 3) 配置环境变量
在项目根目录新建 `.env` 文件，复制以下内容并填写对应配置：

```env
# 必填 - OpenAI API 配置
OPENAI_API_KEY=your-openai-api-key
OPENAI_BASE_URL=your-base-url
OPENAI_MODEL=your-model

# 可选 - 高德API 天气/地理信息（不填则使用模拟数据）
AMAP_API_KEY=your-amap-api-key

# 可选 - 管理员清空口令
PURGE_TOKEN=your-token

# 可选 - 调试模式
DEBUG=true
```

### 4) 运行项目

#### 方式 1：终端交互式使用（推荐）
```bash
python main.py
```

#### 方式 2：启动 API 服务（支持前端对接）
```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

访问：
- 前端页面：http://localhost:8000
- 接口文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health

---

## 常用指令 & 核心接口

### CLI 终端指令
| 指令 | 功能说明 |
|---|---|
| `/state` | 查看当前会话状态（城市 / 天数 / 预算 / 偏好） |
| `/profile` | 查看用户出行偏好画像与记忆数据 |
| `/clear <token>` | 清空所有用户记忆并重置会话 |
| `quit/exit` | 退出终端程序 |


### 核心 API 接口（以 `server.py` 为准）
| 接口地址 | 方法 | 功能描述 |
|---|---|---|
| `/api/chat` | POST | 核心对话接口，生成行程规划 / 穿搭建议 |
| `/health` | GET | 服务健康检查（会验证配置有效性） |
| `/` | GET | 返回内置前端页面（`web/index.html`） |

---

## 技术栈
- 流程编排：LangGraph
- 后端服务：FastAPI
- 大模型能力：OpenAI 兼容 API
- 向量存储：ChromaDB（用户偏好持久化）
- 配置管理：python-dotenv
- 数据来源：高德地图 API（可选）

---

## 项目结构
```text
IntelliGo/
├── main.py           # CLI 终端入口
├── server.py         # FastAPI 服务入口
├── config.py         # 全局配置管理
├── graph/            # LangGraph 流程核心定义
├── memory/           # 用户偏好记忆模块
├── core/             # 业务逻辑封装
├── web/              # 前端页面
├── data/             # 数据持久化目录（建议不要提交到仓库）
├── .env              # 环境变量配置（不要提交到仓库）
└── requirements.txt  # 依赖清单
```

---

<div align="center">
Made with ❤️ 让出行规划更智能、更省心
</div>
