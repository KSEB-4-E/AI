from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
from collections import Counter
import os
import json
import random
import re

# ✅ 환경 변수 로딩 및 GPT 초기화
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ FastAPI 앱 초기화
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 불용어 사전 정의
stopwords = {
    # 조사 및 형식어
    "이", "가", "은", "는", "을", "를", "에", "의", "로", "과", "와", "도", "부터", "까지", "하고", "보다", "든지", "라도", "에서", "조언을",

    # 보조 동사/형용사
    "있다", "있으며" ,"없다", "되다", "하다", "된", "하는", "되고", "되며", "했다", "하는데", "하고", "하게", "하면서", "있는","밝혔다", "만에",

    # 빈도 높은 일반어
    "이번", "검색","지난", "최근", "대해", "대한", "관련", "통해", "가장", "계속", "더욱", "많은", "많이", "또한", "그러나", "그리고", "등", "때문", "때문에", "하지만", "그런", "이런", "저런",

    # 시제/시간/조건
    "지금", "오늘", "내일", "어제", "당시", "현재", "과거", "미래", "앞으로", "이미", "계속", "마지막", "처음", "항상", "때문", "우선",

    # 인칭/지시/수량
    "우리", "당신", "이것", "그것", "저것", "누구", "모두", "각각", "모든", "하나", "둘", "셋",

    # 기타 불필요한 일반어
    "상대", "입장", "정도", "수준", "칠한", "명으로", "사실","주요", "경우", "사람", "부분", "문제", "방법", "상황", "내용", "의견", "주장", "기자", "보도", "뉴스", "정부", "기관", "위원회"
}


# ✅ 간단한 키워드 추출 함수 수정
def extract_keywords_from_text(text: str) -> List[str]:
    words = re.findall(r"[가-힣]{2,5}", text)  # 2~5글자 한글 단어
    return [word for word in words if word not in stopwords]

# ✅ 1. 추천 키워드 API (랜덤 30개 기사 기반)
@app.get("/trending-keywords")
def get_trending_keywords():
    df = pd.read_csv("kobart_news_summarized.csv", encoding="cp949")
    combined_texts = (df["title"].fillna("") + " " + df["summary"].fillna("")).tolist()
    sampled_texts = combined_texts if len(combined_texts) < 30 else random.sample(combined_texts, 30)

    all_keywords = []
    for text in sampled_texts:
        all_keywords += extract_keywords_from_text(text)

    most_common = Counter(all_keywords).most_common(5)
    return {"keywords": [{"keyword": kw, "count": count} for kw, count in most_common]}

# ✅ 2. 기사 검색 API
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

# ✅ 3. 결론 요약 API
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