"""CrossEncoder 精排。向量粗召分不清「雨霖铃」词牌/乐曲/诗句，这里按问句语义压噪声。"""
from __future__ import annotations

from app.config import settings

_reranker = None
_reranker_failed = False


def get_reranker():
    """本地缓存没有模型就跳过精排，不要在对话里现下。先跑 scripts/download_reranker.py。"""
    global _reranker, _reranker_failed
    if _reranker_failed:
        return None
    if _reranker is not None:
        return _reranker
    try:
        from sentence_transformers import CrossEncoder

        device = "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
        except Exception:
            pass
        _reranker = CrossEncoder(
            settings.reranker_model,
            local_files_only=True,
            device=device,
        )
        return _reranker
    except Exception:
        _reranker_failed = True
        print(
            f"重排模型未加载（{settings.reranker_model}），本次只走向量召回。"
            "离线下载：python scripts/download_reranker.py",
            flush=True,
        )
        return None


def rerank_hits(query: str, hits: list[dict], top_k: int) -> list[dict]:
    if not hits:
        return []
    if top_k <= 0:
        return []
    model = get_reranker()
    if model is None or not (query or "").strip():
        return hits[:top_k]
    pairs = [(query, (h.get("text") or "")[:800]) for h in hits]
    try:
        scores = model.predict(pairs, convert_to_numpy=True)
    except Exception:
        return hits[:top_k]
    ranked = list(hits)
    for h, score in zip(ranked, scores):
        h["rerank"] = float(score)
    ranked.sort(key=lambda h: h.get("rerank") or 0.0, reverse=True)
    return ranked[:top_k]
