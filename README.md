# Ollama-OpenAI 接口适配器

这个项目提供了一个代理服务，将 Ollama API 接口转换为兼容 OpenAI API 的格式，使得为 OpenAI API 设计的应用程序可以无缝地使用 Ollama 提供的模型。

## 功能特点

- 支持 OpenAI API 的主要端点：
  - `/v1/models` - 列出可用模型
  - `/v1/chat/completions` - 聊天完成（支持流式响应）
  - `/v1/completions` - 文本完成（支持流式响应）
  - `/v1/embeddings` - 文本嵌入

- 自动转换请求和响应格式
- 支持流式输出（SSE）
- 支持常见参数（温度、top_p、最大令牌数等）

## 安装

1. 克隆此仓库：

```bash
git clone https://github.com/nszy007/o2o.git
cd o2o
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 复制环境变量示例文件并进行配置：

```bash
cp .env.example .env
```

编辑 `.env` 文件，设置 Ollama API 的地址（默认为 `http://localhost:11434`）和服务器的主机/端口。

## 使用方法

### 启动服务器

```bash
python app.py
```

或者使用 uvicorn：

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 使用示例

服务启动后，你可以像使用 OpenAI API 一样使用这个适配器：

#### 列出模型

```bash
curl http://localhost:8000/v1/models
```

#### 聊天完成

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2",
    "messages": [
      {
        "role": "system",
        "content": "你是一个有用的AI助手。"
      },
      {
        "role": "user",
        "content": "你好，能介绍一下你自己吗？"
      }
    ]
  }'
```

#### 流式聊天完成

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2",
    "messages": [
      {
        "role": "user",
        "content": "讲个笑话"
      }
    ],
    "stream": true
  }'
```

#### 文本完成

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2",
    "prompt": "人工智能的未来是",
    "max_tokens": 100
  }'
```

#### 文本嵌入

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2",
    "input": "这是一段需要转换为嵌入向量的文本"
  }'
```

## Ollama 与 OpenAI API 的主要差异

虽然此适配器尽可能地模拟 OpenAI API 的行为，但仍有一些固有的差异：

1. **模型支持**：只能使用 Ollama 中已安装的模型。
2. **参数支持**：部分 OpenAI 高级参数可能不被支持。
3. **嵌入维度**：嵌入向量的维度取决于 Ollama 中使用的模型。
4. **令牌计数**：令牌计数可能与 OpenAI 的计算方式不同。
5. **批量嵌入**：目前 Ollama 不支持批量嵌入，因此嵌入接口当前只处理批量输入的第一项。

## 环境变量

- `OLLAMA_API_BASE_URL`: Ollama API 的基础 URL（默认：`http://localhost:11434`）
- `HOST`: 服务器主机地址（默认：`0.0.0.0`）
- `PORT`: 服务器端口号（默认：`8000`）

## 注意事项

- 此适配器仅提供基本的 API 兼容性，不保证与所有依赖 OpenAI API 的应用程序完全兼容。
- 性能和响应时间可能与 OpenAI API 不同，这取决于你的 Ollama 实例的配置和所使用的模型。
