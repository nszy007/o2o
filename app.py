import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import httpx
import json
import uvicorn
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List, Union
import asyncio

# 加载环境变量
load_dotenv()

# 获取环境变量
OLLAMA_API_BASE_URL = os.getenv("OLLAMA_API_BASE_URL", "http://localhost:11434")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

app = FastAPI(title="Ollama to OpenAI API Adapter")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建一个HTTP客户端
http_client = httpx.AsyncClient(timeout=None)

@app.get("/")
async def root():
    return {"message": "Ollama to OpenAI API Adapter is running"}

@app.get("/v1/models")
async def list_models():
    """列出所有可用模型，转换为OpenAI格式"""
    try:
        response = await http_client.get(f"{OLLAMA_API_BASE_URL}/api/tags")
        ollama_models = response.json().get("models", [])
        
        openai_models = []
        for model in ollama_models:
            openai_models.append({
                "id": model["name"],
                "object": "model",
                "created": 0,
                "owned_by": "ollama",
                "permission": [],
                "root": model["name"],
                "parent": None
            })
            
        return {
            "object": "list",
            "data": openai_models
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """处理聊天完成请求，将OpenAI格式转换为Ollama格式，并返回OpenAI格式的响应"""
    try:
        data = await request.json()
        
        # 从请求中提取必要的信息
        model = data.get("model", "")
        messages = data.get("messages", [])
        stream = data.get("stream", False)
        temperature = data.get("temperature", 0.7)
        top_p = data.get("top_p", 1.0)
        max_tokens = data.get("max_tokens")
        
        # 构建Ollama请求
        ollama_request = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
            }
        }
        
        if max_tokens:
            ollama_request["options"]["num_predict"] = max_tokens
        
        # 发送请求到Ollama API
        if stream:
            return StreamingResponse(
                stream_chat_response(ollama_request),
                media_type="text/event-stream"
            )
        else:
            response = await http_client.post(
                f"{OLLAMA_API_BASE_URL}/api/chat",
                json=ollama_request
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            
            ollama_response = response.json()
            
            # 转换为OpenAI格式
            message = ollama_response.get("message", {})
            
            openai_response = {
                "id": "chatcmpl-" + os.urandom(12).hex(),
                "object": "chat.completion",
                "created": int(asyncio.get_event_loop().time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": message.get("role", "assistant"),
                            "content": message.get("content", "")
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": ollama_response.get("prompt_eval_count", 0),
                    "completion_tokens": ollama_response.get("eval_count", 0),
                    "total_tokens": ollama_response.get("prompt_eval_count", 0) + ollama_response.get("eval_count", 0)
                }
            }
            
            return openai_response
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat completion: {str(e)}")

async def stream_chat_response(ollama_request):
    """流式传输聊天响应"""
    try:
        async with http_client.stream(
            "POST",
            f"{OLLAMA_API_BASE_URL}/api/chat",
            json=ollama_request,
            timeout=None
        ) as response:
            if response.status_code != 200:
                error_detail = await response.text()
                yield f"data: {json.dumps({'error': error_detail})}\n\n"
                return
            
            # 为流式响应创建一个唯一ID
            response_id = "chatcmpl-" + os.urandom(12).hex()
            created_time = int(asyncio.get_event_loop().time())
            
            async for line in response.aiter_lines():
                if not line or line.strip() == "":
                    continue
                
                try:
                    ollama_chunk = json.loads(line)
                    
                    # 判断是否为最终消息
                    is_final = "done" in ollama_chunk and ollama_chunk["done"]
                    
                    # 创建OpenAI格式的块
                    openai_chunk = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": ollama_request["model"],
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop" if is_final else None
                            }
                        ]
                    }
                    
                    # 添加内容（如果有）
                    if "message" in ollama_chunk and "content" in ollama_chunk["message"]:
                        openai_chunk["choices"][0]["delta"] = {
                            "content": ollama_chunk["message"]["content"]
                        }
                    
                    # 第一个块需要包含角色信息
                    if "message" in ollama_chunk and "role" in ollama_chunk["message"]:
                        openai_chunk["choices"][0]["delta"]["role"] = ollama_chunk["message"]["role"]
                    
                    # 如果是最终消息，添加使用统计
                    if is_final and "prompt_eval_count" in ollama_chunk and "eval_count" in ollama_chunk:
                        openai_chunk["usage"] = {
                            "prompt_tokens": ollama_chunk.get("prompt_eval_count", 0),
                            "completion_tokens": ollama_chunk.get("eval_count", 0),
                            "total_tokens": ollama_chunk.get("prompt_eval_count", 0) + ollama_chunk.get("eval_count", 0)
                        }
                    
                    yield f"data: {json.dumps(openai_chunk)}\n\n"
                    
                    # 如果是最终消息，发送完成信号
                    if is_final:
                        yield "data: [DONE]\n\n"
                        break
                        
                except json.JSONDecodeError:
                    continue
                    
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.post("/v1/completions")
async def completions(request: Request):
    """处理文本完成请求，将OpenAI格式转换为Ollama格式，并返回OpenAI格式的响应"""
    try:
        data = await request.json()
        
        # 从请求中提取必要的信息
        model = data.get("model", "")
        prompt = data.get("prompt", "")
        stream = data.get("stream", False)
        temperature = data.get("temperature", 0.7)
        top_p = data.get("top_p", 1.0)
        max_tokens = data.get("max_tokens")
        
        # 构建Ollama请求
        ollama_request = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
            }
        }
        
        if max_tokens:
            ollama_request["options"]["num_predict"] = max_tokens
        
        # 发送请求到Ollama API
        if stream:
            return StreamingResponse(
                stream_completion_response(ollama_request),
                media_type="text/event-stream"
            )
        else:
            response = await http_client.post(
                f"{OLLAMA_API_BASE_URL}/api/generate",
                json=ollama_request
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            
            ollama_response = response.json()
            
            # 转换为OpenAI格式
            openai_response = {
                "id": "cmpl-" + os.urandom(12).hex(),
                "object": "text_completion",
                "created": int(asyncio.get_event_loop().time()),
                "model": model,
                "choices": [
                    {
                        "text": ollama_response.get("response", ""),
                        "index": 0,
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": ollama_response.get("prompt_eval_count", 0),
                    "completion_tokens": ollama_response.get("eval_count", 0),
                    "total_tokens": ollama_response.get("prompt_eval_count", 0) + ollama_response.get("eval_count", 0)
                }
            }
            
            return openai_response
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing completion: {str(e)}")

async def stream_completion_response(ollama_request):
    """流式传输文本完成响应"""
    try:
        async with http_client.stream(
            "POST",
            f"{OLLAMA_API_BASE_URL}/api/generate",
            json=ollama_request,
            timeout=None
        ) as response:
            if response.status_code != 200:
                error_detail = await response.text()
                yield f"data: {json.dumps({'error': error_detail})}\n\n"
                return
            
            # 为流式响应创建一个唯一ID
            response_id = "cmpl-" + os.urandom(12).hex()
            created_time = int(asyncio.get_event_loop().time())
            
            async for line in response.aiter_lines():
                if not line or line.strip() == "":
                    continue
                
                try:
                    ollama_chunk = json.loads(line)
                    
                    # 判断是否为最终消息
                    is_final = "done" in ollama_chunk and ollama_chunk["done"]
                    
                    # 创建OpenAI格式的块
                    openai_chunk = {
                        "id": response_id,
                        "object": "text_completion.chunk",
                        "created": created_time,
                        "model": ollama_request["model"],
                        "choices": [
                            {
                                "text": ollama_chunk.get("response", ""),
                                "index": 0,
                                "finish_reason": "stop" if is_final else None
                            }
                        ]
                    }
                    
                    # 如果是最终消息，添加使用统计
                    if is_final and "prompt_eval_count" in ollama_chunk and "eval_count" in ollama_chunk:
                        openai_chunk["usage"] = {
                            "prompt_tokens": ollama_chunk.get("prompt_eval_count", 0),
                            "completion_tokens": ollama_chunk.get("eval_count", 0),
                            "total_tokens": ollama_chunk.get("prompt_eval_count", 0) + ollama_chunk.get("eval_count", 0)
                        }
                    
                    yield f"data: {json.dumps(openai_chunk)}\n\n"
                    
                    # 如果是最终消息，发送完成信号
                    if is_final:
                        yield "data: [DONE]\n\n"
                        break
                        
                except json.JSONDecodeError:
                    continue
                    
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.post("/v1/embeddings")
async def embeddings(request: Request):
    """处理嵌入请求，将OpenAI格式转换为Ollama格式，并返回OpenAI格式的响应"""
    try:
        data = await request.json()
        
        # 从请求中提取必要的信息
        model = data.get("model", "")
        input_text = data.get("input", "")
        
        # 如果输入是列表，只处理第一项（Ollama目前不支持批量嵌入）
        if isinstance(input_text, list):
            if not input_text:
                raise HTTPException(status_code=400, detail="Input list cannot be empty")
            first_input = input_text[0]
        else:
            first_input = input_text
            input_text = [first_input]  # 转换为列表以保持一致的输出格式
        
        # 构建Ollama请求
        ollama_request = {
            "model": model,
            "prompt": first_input
        }
        
        # 发送请求到Ollama API
        response = await http_client.post(
            f"{OLLAMA_API_BASE_URL}/api/embeddings",
            json=ollama_request
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        
        ollama_response = response.json()
        
        # 转换为OpenAI格式
        openai_response = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": ollama_response.get("embedding", [])
                }
            ],
            "model": model,
            "usage": {
                "prompt_tokens": len(first_input.split()),
                "total_tokens": len(first_input.split())
            }
        }
        
        return openai_response
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing embeddings: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT) 