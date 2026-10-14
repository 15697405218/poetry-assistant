from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class Poem(Base):
    __tablename__ = "poems"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    author: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    dynasty: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    body: Mapped[str] = mapped_column(Text, nullable=False)
    document: Mapped[str] = mapped_column(Text, nullable=False)
    title_aliases: Mapped[str] = mapped_column(Text, nullable=False, default="")
    source_path: Mapped[str] = mapped_column(Text, nullable=False, default="")

    keys: Mapped[list["PoemKey"]] = relationship(back_populates="poem", cascade="all, delete-orphan")


class PoemKey(Base):
    """持久化倒排：首句 / 标题 / 一作 → 诗。"""
    __tablename__ = "poem_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    prefix: Mapped[str] = mapped_column(String(8), nullable=False, default="", index=True)
    poem_id: Mapped[int] = mapped_column(ForeignKey("poems.id", ondelete="CASCADE"), nullable=False, index=True)

    poem: Mapped[Poem] = relationship(back_populates="keys")
