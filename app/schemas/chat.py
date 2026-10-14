from datetime import datetime
from pydantic import BaseModel

class ConversationCreate(BaseModel):
    title: str = "新对话"

class ConversationOut(BaseModel):
    id: int
    title: str
    created_at: datetime
    model_config = {"from_attributes": True}

class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ChatIn(BaseModel):
    conversation_id: int
    content: str