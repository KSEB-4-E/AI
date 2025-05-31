from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import os
import json
import random
from summa import keywords as summa_keywords

# 환경 변수 로딩 및 GPT 초기화
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# FastAPI 앱 초기화
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. 추천 키워드 API (TextRank 기반)
@app.get("/trending-keywords")
def get_trending_keywords():
    df = pd.read_csv("kobart_news_summarized.csv", encoding="cp949")

    combined_texts = (df["title"].fillna("") + " " + df["summary"].fillna("")).tolist()
    if len(combined_texts) < 30:
        sampled_texts = combined_texts
    else:
        sampled_texts = random.sample(combined_texts, 30)

    text = " ".join(sampled_texts)

    # ✅ split=True: 단어 리스트만 반환 (점수 없음)
    extracted = summa_keywords.keywords(text, words=30, split=True)

    # ✅ 두 글자 이상 필터링 & 상위 5개 추출
    filtered_keywords = [kw for kw in extracted if len(kw) >= 2][:5]

    return {
        "keywords": [{"keyword": kw} for kw in filtered_keywords]
    }

# 2. 기사 검색 API
@app.get("/search-articles")
def search_articles(keyword: str = Query(..., min_length=2)):
    df = pd.read_csv("kobart_news_summarized.csv", encoding="cp949")
    filtered = df[
        df["title"].fillna("").str.contains(keyword, case=False, regex=False) |
        df["summary"].fillna("").str.contains(keyword, case=False, regex=False)
    ].copy()
    filtered["row_order"] = filtered.index[::-1]
    latest_by_source = (
        filtered.sort_values("row_order")
        .drop_duplicates(subset=["source"], keep="first")
    )
    latest_articles = latest_by_source.sort_values("row_order").head(3)
    articles = latest_articles[["title", "summary", "content", "source", "link"]].to_dict(orient="records")
    return {"keyword": keyword, "articles": articles}

# 3. 결론 요약 API
class SummaryRequest(BaseModel):
    keyword: str
    contents: List[str]

@app.post("/summarize-conclusion")
def summarize_conclusion(data: SummaryRequest):
    keyword = data.keyword
    contents = data.contents[:3]
    joined_content = "\n".join(contents)
    prompt = f"""
다음은 '{keyword}'에 대한 여러 언론사의 기사 원문입니다.

이 기사들의 공통된 주제를 다음 3가지 항목으로 간결히 정리해줘.

요약 형식은 다음 JSON 형태 그대로 출력해줘:

{{
  "fact": "핵심 사실을 요약",
  "issue": "신문사들의 공통 쟁점 요약",
  "outlook": "향후 전망 또는 종합 판단 요약"
}}

조건:
- 각 항목은 1문장으로 요약
- 직접 인용 없이 요점을 명확히 서술
- 항목 이름은 반드시 "fact", "issue", "outlook"으로 유지

기사 원문:
{joined_content}
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=500
    )
    content = response.choices[0].message.content.strip()
    try:
        summary_json = json.loads(content)
    except json.JSONDecodeError:
        summary_json = {
            "fact": "요약 실패",
            "issue": "요약 실패",
            "outlook": "요약 실패"
        }
    return {"keyword": keyword, "summary": summary_json}

# uvicorn summarize_conclusion_csv:app --reload
# 키워드: http://localhost:8000/trending-keywords
#
# 본문검색: http://localhost:8000/search-articles?keyword=카카오
#
# http://localhost:8000/docs