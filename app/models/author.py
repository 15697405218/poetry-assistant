from sqlalchemy import Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class Author(Base):
    """作者简介。精确按 name 取，不走向量。"""

    __tablename__ = "authors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    bio: Mapped[str] = mapped_column(Text, nullable=False, default="")
    source_path: Mapped[str] = mapped_column(Text, nullable=False, default="")
