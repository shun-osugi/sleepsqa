from fastapi import FastAPI
from pydantic import BaseModel
from annealing import simulated_annealing  # ここでアニーリングロジックをインポート

app = FastAPI()

# 入力データの構造
class SleepRequest(BaseModel):
    cannot_sleep: list[list[bool]]
    target_sleep_time: int
    ideal_segments: int

@app.post("/sleep_schedule/")
async def calculate_sleep_schedule(request: SleepRequest):
    # アニーリングのロジックを呼び出す
    schedule = simulated_annealing(request.cannot_sleep, request.target_sleep_time, request.ideal_segments)
    
    # 計算結果を返す
    return {"schedule": schedule.tolist()}  # NumPyの配列をリストに変換して返す
