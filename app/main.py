from fastapi import FastAPI
from pydantic import BaseModel
from .graph import hr_app, HRState

app = FastAPI(title="HR Workflow Agent")

class ChatRequest(BaseModel):
    user_id: str | None = None
    message: str

class ChatResponse(BaseModel):
    route: str
    answer: str

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    initial: HRState = {
        "user_input": req.message,
    }
    if req.user_id:
        initial["user_id"] = req.user_id

    final_state = hr_app.invoke(initial)
    return ChatResponse(
        route=final_state["intent"],
        answer=final_state["answer"],
    )
