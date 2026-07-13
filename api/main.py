import os
import json
import time
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from api.roles import ROLES

load_dotenv()

app = FastAPI(title="松果AI解压伙伴")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DeepSeek客户端
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 会话存储
sessions: Dict[str, List[Dict]] = {}

class ChatRequest(BaseModel):
    message: str
    role: str = "松果"
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    role: str
    session_id: str

@app.get("/")
def root():
    return {"message": "松果AI解压伙伴 API", "status": "running"}

@app.get("/api/roles")
def get_roles():
    return {
        name: {"name": info["name"], "scene": info.get("scene", ""), "opening": info["opening"]}
        for name, info in ROLES.items()
    }

@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        # 验证角色
        if request.role not in ROLES:
            raise HTTPException(status_code=400, detail=f"未知角色: {request.role}")
        
        # 生成或使用session_id
        session_id = request.session_id or f"session_{int(time.time())}"
        
        # 初始化会话
        if session_id not in sessions:
            sessions[session_id] = []
        
        # 添加用户消息
        sessions[session_id].append({"role": "user", "content": request.message})
        
        # 构建消息列表
        messages = [
            {"role": "system", "content": ROLES[request.role]["system_prompt"]}
        ] + sessions[session_id]
        
        # 调用DeepSeek API
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.7,
            max_tokens=1024
        )
        
        reply = response.choices[0].message.content
        
        # 添加助手回复
        sessions[session_id].append({"role": "assistant", "content": reply})
        
        # 限制会话长度
        if len(sessions[session_id]) > 20:
            sessions[session_id] = sessions[session_id][-20:]
        
        return ChatResponse(
            reply=reply,
            role=request.role,
            session_id=session_id
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reset")
def reset_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
