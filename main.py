import asyncio
import uuid
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from rag_engine import ask, ask_stream

app = FastAPI()

class ChatRequest(BaseModel):
    question: str
    session_id: str = ""

class ChatResponse(BaseModel):
    answer: str
    session_id: str

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(None, ask, req.question, session_id)
    return ChatResponse(answer=answer, session_id=session_id)

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())

    async def generate():
        async for token in ask_stream(req.question, session_id):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"X-Session-Id": session_id},
    )

@app.get("/health")
async def health():
    return {"status": "ok"}
