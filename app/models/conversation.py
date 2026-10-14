
from datetime import datetime
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base

class Conversation(Base):
    __tablename__ = "conversations"

    id:Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id:Mapped[int] = mapped_column(ForeignKey("users.id"), index=True,nullable=False)
    title:Mapped[str] = mapped_column(String(255), nullable=False)
    summary:Mapped[str | None] = mapped_column(Text, nullable=True)
    summarized_through_id:Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at:Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(), 
    )
    messages: Mapped[list["Message"]] = relationship(back_populates="conversation")
