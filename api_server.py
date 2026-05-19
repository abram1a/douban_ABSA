# -*- coding: utf-8 -*-
"""
豆瓣电影评论多维度情感分析 - HTTP API 服务
===========================================

启动方式：
    pip install fastapi uvicorn pydantic         # 首次需要安装
    python api_server.py                          # 启动服务

启动后访问：
    交互式文档:    http://127.0.0.1:8000/docs
    单条预测:      POST http://127.0.0.1:8000/predict     {"text": "演技很棒"}
    批量预测:      POST http://127.0.0.1:8000/predict_batch {"texts": ["...", "..."]}
    简化输出:      POST http://127.0.0.1:8000/predict_simple {"text": "..."}

调用示例（curl）：
    curl -X POST http://127.0.0.1:8000/predict \\
         -H "Content-Type: application/json" \\
         -d '{"text":"演技太棒了但剧情拖沓"}'

调用示例（Python requests）：
    import requests
    r = requests.post("http://127.0.0.1:8000/predict",
                      json={"text": "演技太棒了但剧情拖沓"})
    print(r.json())
"""
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from predictor import Predictor

# ════════════════════════════════════════════════════════════
# 初始化
# ════════════════════════════════════════════════════════════
app = FastAPI(
    title="豆瓣电影评论多维度情感分析 API",
    description="输入中文评论，返回演员/剧情/特效/音乐/导演 5 个维度的情感预测",
    version="1.0",
)

# 允许浏览器跨域访问（前端 HTML 调用必需）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 服务启动时一次性加载模型（不会在每次请求时重新加载）
print("[API] 正在加载模型...")
predictor = Predictor(verbose=True)
print("[API] 模型加载完成，服务就绪")


# ════════════════════════════════════════════════════════════
# 请求/响应数据结构
# ════════════════════════════════════════════════════════════
class PredictRequest(BaseModel):
    text: str = Field(..., description="评论文本", example="演技太棒了但剧情拖沓")
    top_k_words: int = Field(3, ge=0, le=10, description="每维度返回的关注词数量")


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., description="评论文本列表",
                             example=["剧情很赞", "特效一般"])
    top_k_words: int = Field(3, ge=0, le=10)


# ════════════════════════════════════════════════════════════
# 路由
# ════════════════════════════════════════════════════════════
@app.get("/")
def root():
    """根路径，返回服务信息"""
    return {
        "service": "豆瓣多维度情感分析 API",
        "endpoints": {
            "POST /predict":         "单条详细预测",
            "POST /predict_batch":   "批量预测",
            "POST /predict_simple":  "单条简化预测（只返回命中维度）",
            "GET  /health":          "健康检查",
            "GET  /docs":            "Swagger 交互文档",
        },
    }


@app.get("/health")
def health():
    """健康检查（监控用）"""
    return {"status": "ok", "model_loaded": predictor is not None}


@app.post("/predict")
def predict_endpoint(req: PredictRequest):
    """
    单条评论的详细预测。

    返回：
    - text: 原文本
    - dimensions: 5 个维度的详细结果（命中/未命中、情感、置信度、关注词）
    """
    try:
        return predictor.predict(req.text, top_k_words=req.top_k_words)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {e}")


@app.post("/predict_batch")
def predict_batch_endpoint(req: BatchPredictRequest):
    """批量预测，一次最多 100 条"""
    if len(req.texts) > 100:
        raise HTTPException(status_code=400, detail="单次最多 100 条")
    try:
        return {"results": predictor.predict_batch(req.texts,
                                                   top_k_words=req.top_k_words)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {e}")


@app.post("/predict_simple")
def predict_simple_endpoint(req: PredictRequest):
    """
    简化预测：只返回命中的维度及情感（适合快速展示）。

    示例返回：
        {"text": "演技好但剧情烂",
         "results": {"演员": "正面", "剧情": "负面"}}
    """
    try:
        return predictor.predict_simple(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {e}")


# ════════════════════════════════════════════════════════════
# 启动
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
