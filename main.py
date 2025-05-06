from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import httpx
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

# Load biến môi trường
load_dotenv()
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("Thiếu API_KEY trong file .env")

GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={API_KEY}"

app = FastAPI()

# Cho phép CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Nếu có domain thì thay vào
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = joblib.load("model.pkl")

# Schema input từ landmark
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

# Prompt cố định
INSTRUCTION = (
    "Bạn là một con bot hỗ trợ đưa ra câu văn dựa trên các keyword mà người dùng truyền vào.\n"
    "1. Hãy dựa vào những keyword này để đưa ra câu hoàn chỉnh.\n"
    "2. Không được phép lan man và chỉ được đưa ra câu trả lời thôi.\n"
    "3. Ví dụ người dùng đưa ra là: tôi, học, trường, xe thì câu trả lời chỉ được phép đưa ra là: Tôi đi học đến trường bằng xe đạp. Chỉ có vậy thôi.\n"
)

# Schema keywords
class ChatRequest(BaseModel):
    keywords: str

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

    async with httpx.AsyncClient() as client:
        response = await client.post(GEMINI_ENDPOINT, json=payload)

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    try:
        data = response.json()
        reply = data["candidates"][0]["content"]["parts"][0]["text"]
        return {"message": reply}
    except Exception:
        raise HTTPException(status_code=500, detail="Lỗi xử lý phản hồi từ Gemini")
