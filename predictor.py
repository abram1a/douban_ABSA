# -*- coding: utf-8 -*-
"""
豆瓣电影评论多维度情感分析 - 预测接口
===========================================

提供两种使用方式：

【方式 1】函数式（一次性快速调用）：
    from predictor import predict
    result = predict("演技太棒了但剧情很烂")
    print(result)

【方式 2】类式（推荐，模型只加载一次，批量预测更快）：
    from predictor import Predictor
    p = Predictor(model_dir="outputs_v6")
    print(p.predict("演技太棒了但剧情很烂"))
    print(p.predict_batch(["剧情拖沓", "配乐很赞"]))

返回结果格式（字典）：
{
    "text": "原文本",
    "dimensions": {
        "演员": {"present": True, "sentiment": "正面", "prob": 0.92, "top_words": ["演技", "棒"]},
        "剧情": {"present": True, "sentiment": "负面", "prob": 0.78, "top_words": ["剧情", "烂"]},
        "特效": {"present": False, "prob": 0.05},
        "音乐": {"present": False, "prob": 0.03},
        "导演": {"present": False, "prob": 0.12}
    }
}
"""
import numpy as np
import torch
from v17 import (
    _load_model_for_infer, _predict_one,
    ASPECTS, ASP_CN, SENT_CN, OUT_DIR,
)


class Predictor:
    """情感预测器（推荐使用，模型只加载一次）"""

    def __init__(self, model_dir=OUT_DIR, verbose=False):
        """
        加载模型和词表。
        Args:
            model_dir: 训练输出目录，默认 outputs_v6
            verbose: 是否打印加载日志
        """
        if verbose:
            print(f"[Predictor] 加载模型: {model_dir}")
        self.model, self.vocab = _load_model_for_infer(model_dir)
        self.model.eval()
        if verbose:
            print(f"[Predictor] 模型加载完成")

    def predict(self, text: str, top_k_words: int = 3) -> dict:
        """
        预测单条评论的多维度情感。

        Args:
            text: 评论文本
            top_k_words: 每个维度返回的注意力 top-k 词数量（默认 3）

        Returns:
            dict: 包含 text 和 dimensions（5 个维度的情感与关注词）
        """
        if not text or not str(text).strip():
            return {"text": text, "dimensions": {},
                    "error": "输入为空"}

        res, attn_w, toks = _predict_one(str(text), self.model, self.vocab)

        dimensions = {}
        for ai, asp in enumerate(ASPECTS):
            info = res[asp]
            asp_cn = ASP_CN[asp]
            if info["present"]:
                # 注意力 top-k 词
                attn = np.asarray(attn_w[ai])
                top_idx = attn.argsort()[::-1][:top_k_words]
                top_words = [toks[j] for j in top_idx if j < len(toks)]
                dimensions[asp_cn] = {
                    "present": True,
                    "sentiment": info["sentiment"],
                    "sentiment_id": info["sentiment_id"],
                    "prob": round(float(info["prob"]), 4),
                    "top_words": top_words,
                }
            else:
                dimensions[asp_cn] = {
                    "present": False,
                    "prob": round(float(info["prob"]), 4),
                }

        return {"text": text, "dimensions": dimensions}

    def predict_batch(self, texts: list, top_k_words: int = 3) -> list:
        """
        批量预测（仅是循环调用 predict，方便接入）。
        """
        return [self.predict(t, top_k_words=top_k_words) for t in texts]

    def predict_simple(self, text: str) -> dict:
        """
        简化输出：只返回命中的维度及其情感（适合快速展示）。
        Returns: {"text": ..., "results": {"演员": "正面", "剧情": "负面"}}
        """
        full = self.predict(text)
        simple = {asp_cn: info["sentiment"]
                  for asp_cn, info in full["dimensions"].items()
                  if info.get("present")}
        return {"text": text, "results": simple}


# ════════════════════════════════════════════════════════════
# 全局便捷接口（懒加载，第一次调用时才加载模型）
# ════════════════════════════════════════════════════════════
_GLOBAL_PREDICTOR = None


def predict(text: str, model_dir: str = OUT_DIR, top_k_words: int = 3) -> dict:
    """
    一行式预测接口。第一次调用会加载模型（较慢），之后调用都会复用。

    Example:
        >>> from predictor import predict
        >>> predict("演技很棒但剧情拖沓")
    """
    global _GLOBAL_PREDICTOR
    if _GLOBAL_PREDICTOR is None:
        _GLOBAL_PREDICTOR = Predictor(model_dir=model_dir, verbose=True)
    return _GLOBAL_PREDICTOR.predict(text, top_k_words=top_k_words)


def predict_batch(texts: list, model_dir: str = OUT_DIR, top_k_words: int = 3) -> list:
    """批量版"""
    global _GLOBAL_PREDICTOR
    if _GLOBAL_PREDICTOR is None:
        _GLOBAL_PREDICTOR = Predictor(model_dir=model_dir, verbose=True)
    return _GLOBAL_PREDICTOR.predict_batch(texts, top_k_words=top_k_words)


# ════════════════════════════════════════════════════════════
# 命令行测试
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import json
    p = Predictor(verbose=True)

    test_cases = [
        "演技太棒了，但剧情有点拖沓，特效做得很用心",
        "配乐很加分，导演手法成熟",
        "完全是圈钱之作，毫无诚意",
        "还行吧",
    ]

    print("\n" + "=" * 60)
    print("详细预测结果（含注意力关注词）")
    print("=" * 60)
    for text in test_cases:
        result = p.predict(text)
        print(f"\n输入: {result['text']}")
        for asp_cn, info in result["dimensions"].items():
            if info["present"]:
                top = "  关注词: " + ", ".join(info["top_words"]) if info.get("top_words") else ""
                print(f"  {asp_cn:4s}  {info['sentiment']}  (置信度 {info['prob']:.0%}){top}")
            else:
                print(f"  {asp_cn:4s}  未提及  (检测概率 {info['prob']:.0%})")

    print("\n" + "=" * 60)
    print("简化输出示例（只显示命中维度）")
    print("=" * 60)
    for text in test_cases:
        result = p.predict_simple(text)
        print(f"\n  {result['text']}")
        print(f"  → {result['results']}")

    print("\n" + "=" * 60)
    print("JSON 输出示例（适合传给前端 / API）")
    print("=" * 60)
    print(json.dumps(p.predict(test_cases[0]), ensure_ascii=False, indent=2))
