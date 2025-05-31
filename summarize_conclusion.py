from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
import os

# ✅ 환경 변수 로딩 및 GPT 클라이언트 초기화
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ FastAPI 서버 초기화
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프론트 연결 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 요청 바디 모델 정의
class SummaryRequest(BaseModel):
    keyword: str
    contents: List[str]  # 기사 본문 리스트

# ✅ 결론 요약 API
@app.post("/summarize-conclusion")
def summarize_conclusion(data: SummaryRequest):
    keyword = data.keyword
    contents = data.contents[:3]  # 최대 3개 기사까지만 사용

    prompt = f"""
다음은 '{keyword}'에 대한 여러 언론사의 기사 원문입니다.

이 기사들의 핵심 내용을 바탕으로 전체 이슈의 흐름을 3문장 이내로 요약해줘.

- 요약은 전체 기사 내용의 약 5~10% 정도 압축된 분량이길 바라.
- 각각의 문장은 (1) 핵심 사실, (2) 언론 입장 요약, (3) 향후 전망 또는 종합 판단 을 담되, 중립적인 시선으로 재구성해줘.
- 직접적인 인용보다는 개념을 정리해서 서술해줘.

""" + "\n\n".join([f"기사 {i+1}:\n{c}" for i, c in enumerate(contents)])

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=500
    )

    return {
        "keyword": keyword,
        "summary": response.choices[0].message.content.strip()
    }