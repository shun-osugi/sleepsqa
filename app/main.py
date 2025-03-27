from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from annealing import simulated_annealing  # ここでアニーリングロジックをインポート
from fastapi.middleware.cors import CORSMiddleware
import numpy as np


app = FastAPI()

# CORS設定（Flutterからのリクエストを許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # セキュリティを考慮するなら適切なドメインを指定
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 入力データの構造
class SleepRequest(BaseModel):
    cannot_sleep: list[list[bool]]
    target_sleep_time: int
    ideal_segments: int

@app.post("/sleep_schedule/")
async def calculate_sleep_schedule(request: SleepRequest):
    try:
        # アニーリングのロジックを呼び出す
        schedule = simulated_annealing(request.cannot_sleep, request.target_sleep_time, request.ideal_segments)
        
        # NumPy配列をリストに変換し、すべての値をintにキャスト
        if isinstance(schedule, np.ndarray):  
            schedule = schedule.astype(int).tolist()

        return {"schedule": schedule}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))