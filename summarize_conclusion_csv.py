from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os
import json
import random
import re
from konlpy.tag import Okt
import openai

# 환경 변수 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# FastAPI 초기화
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TF-IDF 기반 키워드 추출 함수 (Okt 기반 명사 추출 및 조사 제거)
def extract_keywords(texts, top_n=5):
    stopwords = set([
        "그리고", "그러나", "하지만", "또한", "등", "이", "그", "저", "것", "수",
        "명이", "으로", "명으로", "들", "에서", "하다", "한", "대해", "있다", "대한",
        "밝혔다", "후보는", "칠한", "지난", "있는", "주요", "로", "은", "는", "이", "가",
        "을", "를", "에", "의", "와", "과", "도"
    ])

    try:
        okt = Okt()
        tokenized = []
        for text in texts:
            nouns = [word for word in okt.nouns(text) if len(word) > 1 and word not in stopwords]
            tokenized.append(" ".join(nouns))

        if not any(tokenized):
            return []

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(tokenized)
        feature_names = vectorizer.get_feature_names_out()
        scores = X.sum(axis=0).A1

        keywords = [(feature_names[i], round(scores[i])) for i in range(len(scores))]
        keywords.sort(key=lambda x: x[1], reverse=True)
        return [{"keyword": kw, "count": int(count)} for kw, count in keywords[:top_n]]
    except Exception as e:
        print("TF-IDF 키워드 추출 오류:", e)
        return []

@app.get("/trending-keywords")
def get_trending_keywords():
    try:
        df = pd.read_csv("kobart_news_summarized.csv", encoding="cp949")
        combined = (df["title"].fillna("") + " " + df["summary"].fillna(""))
        sampled_texts = random.sample(combined.tolist(), min(30, len(combined)))
        keywords = extract_keywords(sampled_texts, top_n=5)
        return {"keywords": keywords}
    except Exception as e:
        return {"error": str(e)}

@app.get("/search-articles")
def search_articles(keyword: str = Query(..., min_length=2)):
    try:
        df = pd.read_csv("kobart_news_summarized.csv", encoding="cp949")
        filtered = df[
            df["title"].fillna("").str.contains(keyword, case=False, regex=False) |
            df["summary"].fillna("").str.contains(keyword, case=False, regex=False)
        ].copy()
        filtered["row_order"] = filtered.index[::-1]
        latest_by_source = filtered.sort_values("row_order").drop_duplicates(subset=["source"], keep="first")
        latest_articles = latest_by_source.sort_values("row_order").head(3)
        articles = latest_articles[["title", "summary", "content", "source", "link"]].to_dict(orient="records")
        return {"keyword": keyword, "articles": articles}
    except Exception as e:
        return {"error": str(e)}

# ✅ 3. 결론 요약 API
class SummaryRequest(BaseModel):
    keyword: str
    contents: List[str]

@app.post("/summarize-conclusion")
def summarize_conclusion(data: SummaryRequest):
    keyword = data.keyword
    contents = data.contents[:3]
    context = "\n".join(contents)

    prompt = f"""
다음은 '{keyword}'에 대한 여러 언론사의 기사 원문입니다.

각 기사에서 공통적으로 다루는 이슈의 핵심을 다음 세 항목으로 요약해줘. 항목마다 한 문장 이내로 명확하게 설명해줘.

1. 핵심 사실
2. 각 신문사 기사들의 공통된 쟁점
3. 향후 전망 또는 종합 판단

요약 형식은 다음 JSON 형태로 출력해줘:
{{
  "fact": "...",
  "issue": "...",
  "outlook": "..."
}}

조건:
- 각 항목은 반드시 1문장으로 작성
- 직접 인용 없이 핵심 요점을 명확히 정리
- JSON 구조를 반드시 유지

기사 원문:
{context}
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=500
        )
        content = response.choices[0].message["content"].strip()
        summary_json = json.loads(content)

        return {"keyword": keyword, "summary": summary_json}

    except Exception as e:
        print("결론 요약 오류:", e)
        return {
            "keyword": keyword,
            "summary": {
                "fact": "요약 실패",
                "issue": "요약 실패",
                "outlook": "요약 실패",
                "error": str(e)
            }
        }

# uvicorn summarize_conclusion_csv:app --reload
# 키워드: http://localhost:8000/trending-keywords
# 본문검색: http://localhost:8000/search-articles?keyword=카카오
# 문서화: http://localhost:8000/docs