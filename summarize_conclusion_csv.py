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

# TF-IDF 기반 키워드 추출 함수 (Okt 기반 명사 추출)
def extract_keywords(texts, top_n=5):
    stopwords = set([...])  # 기존과 동일

    try:
        # 정규표현식 기반으로 2글자 이상 한글만 추출
        tokenized = []
        for text in texts:
            words = re.findall(r"[가-힣]{2,}", text)
            words = [w for w in words if w not in stopwords]
            tokenized.append(" ".join(words))

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
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=500
        )
        content = response.choices[0].message["content"].strip()
        summary_json = json.loads(content)
    except Exception as e:
        summary_json = {
            "fact": "요약 실패",
            "issue": "요약 실패",
            "outlook": "요약 실패",
            "error": str(e)
        }

    return {"keyword": keyword, "summary": summary_json}

# uvicorn summarize_conclusion_csv:app --reload
# 키워드: http://localhost:8000/trending-keywords
# 본문검색: http://localhost:8000/search-articles?keyword=카카오
# 문서화: http://localhost:8000/docs