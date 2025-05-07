from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import httpx
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import asyncio

# Load biến môi trường
load_dotenv()
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("Thiếu API_KEY trong file .env")

GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite-001:generateContent?key={API_KEY}"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load mô hình
model = joblib.load("model.pkl")

class LandmarkInput(BaseModel):
    input: list[float]

@app.post("/predict")
async def predict(data: LandmarkInput):
    try:
        features = np.array(data.input).reshape(1, -1)
        prediction = model.predict(features)
        return {"result": str(prediction[0])}
    except Exception as e:
        return {"error": str(e)}

# Câu lệnh hướng dẫn
INSTRUCTION = (
    "Bạn là một con bot hỗ trợ đưa ra câu văn dựa trên các keyword mà người dùng truyền vào.\n"
    "1. Hãy dựa vào những keyword này để đưa ra 3 opiton câu hoàn chỉnh để người dùng có thể lựa chọn.\n"
    "2. Không được phép lan man và chỉ được đưa ra câu trả lời thôi. sau mỗi một câu đưa ra hãy để || để phân biệt các câu \n"
    "3. Ví dụ người dùng đưa ra là: tôi, học, trường, xe thì câu trả lời chỉ được phép đưa ra là: Tôi đi học đến trường bằng xe đạp. Chỉ có vậy thôi.\n"
)

class ChatRequest(BaseModel):
    keywords: str

# Hàm retry nếu lỗi 429
async def send_with_retry(payload, retries=3, backoff_base=1.5):
    async with httpx.AsyncClient() as client:
        for attempt in range(retries):
            response = await client.post(GEMINI_ENDPOINT, json=payload)

            if response.status_code == 200:
                return response.json()

            if response.status_code == 429:
                wait_time = backoff_base ** attempt
                print(f"Bị lỗi 429, đang chờ {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
            else:
                raise HTTPException(status_code=response.status_code, detail=response.text)

        raise HTTPException(status_code=429, detail="Quá nhiều yêu cầu, hãy thử lại sau vài phút.")

@app.post("/gpt/send-final")
async def interpret_final_results(data: dict):
    print("Dữ liệu nhận được từ frontend:", data)

    final_results = data.get("keywords", "").split(', ')
    if not final_results or final_results == ['']:
        raise HTTPException(status_code=400, detail="Không có dữ liệu gửi lên.")

    keywords = ", ".join(final_results)
    full_prompt = f"{INSTRUCTION}Keyword: {keywords}"

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": full_prompt}]
            }
        ]
    }

    response_data = await send_with_retry(payload)

    try:
        reply = response_data["candidates"][0]["content"]["parts"][0]["text"]
        return {"message": reply}
    except Exception:
        raise HTTPException(status_code=500, detail="Lỗi xử lý phản hồi từ Gemini")
