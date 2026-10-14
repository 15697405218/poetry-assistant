
from datetime import datetime
from sqlalchemy import DateTime, ForeignKey, Integer, String,func,Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base

class Message(Base):
    __tablename__ = "messages"

    id:Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    conversation_id:Mapped[int] = mapped_column(ForeignKey("conversations.id"), index=True,nullable=False)
    content:Mapped[str] = mapped_column(Text, nullable=False)
    rewrite:Mapped[str | None] = mapped_column(Text, nullable=True)
    role:Mapped[str] = mapped_column(String(32), nullable=False)
    created_at:Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(), 
    )
    conversation:Mapped["Conversation"] = relationship(back_populates="messages")