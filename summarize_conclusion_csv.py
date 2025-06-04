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

# TF-IDF 기반 키워드 추출 함수 (정규표현식 기반 명사 필터링)
def extract_keywords(texts, top_n=5):
    stopwords = set([
        "그리고", "그러나", "하지만", "또한", "등", "이", "그", "저", "것", "수",
        "명이", "으로", "명으로", "들", "에서", "하다", "한", "대해", "있다", "대한",
        "밝혔다", "후보는", "칠한", "지난", "있는", "주요", "로", "은", "는", "이", "가",
        "을", "를", "에", "의", "와", "과", "도", "것으로", "가운데", "대통령은", "나눔의", "대통령이", "물론", "되겠다",
        "만에", "내일", "당신의", "기사를", "동향과", "정부의"
    ])

    try:
        # 전처리: 불용어 제거
        tokenized = []
        for text in texts:
            words = re.findall(r"[가-힣]{2,}", text)
            words = [w for w in words if w not in stopwords]
            tokenized.append(" ".join(words))

        if not any(tokenized):
            return []

        # TF-IDF 벡터라이저
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(tokenized)
        feature_names = vectorizer.get_feature_names_out()

        # 단어별 TF-IDF 총합 및 등장 횟수 계산
        tfidf_scores = X.sum(axis=0).A1
        word_counts = (X > 0).sum(axis=0).A1  # 단어 등장 문서 수 (or 사용하고 싶으면 `.toarray().sum(axis=0)` 으로 전체 등장 횟수)

        keywords = []
        for idx, word in enumerate(feature_names):
            if word not in stopwords:
                keywords.append((word, int(word_counts[idx])))

        keywords.sort(key=lambda x: x[1], reverse=True)
        return [{"keyword": kw, "count": count} for kw, count in keywords[:top_n]]

    except Exception as e:
        print("TF-IDF 키워드 추출 오류:", e)
        return []

@app.get("/trending-keywords")
def get_trending_keywords():
    try:
        df = pd.read_excel("kobart_news_summarized.xlsx")
        combined = (df["title"].fillna("") + " " + df["summary"].fillna(""))
        sampled_texts = random.sample(combined.tolist(), min(30, len(combined)))
        keywords = extract_keywords(sampled_texts, top_n=5)
        return {"keywords": keywords}
    except Exception as e:
        return {"error": str(e)}

@app.get("/search-articles")
def search_articles(keyword: str = Query(..., min_length=2)):
    try:
        df = pd.read_excel("kobart_news_summarized.xlsx")
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

이 기사들의 공통된 주제를 다음 3가지 항목으로 간결히 정리해줘.

요약 형식은 다음 JSON 형태 그대로 출력해줘:
{{
  "fact": "핵심 사실을 1문장으로 요약",
  "issue": "신문사들의 공통된 쟁점을 1문장으로 요약",
  "outlook": "향후 전망 또는 종합 판단을 1문장으로 요약"
}}

조건:
- 각 항목은 반드시 1문장
- 직접 인용 없이 요점을 명확히 서술
- 항목 이름은 반드시 "fact", "issue", "outlook"만 사용
- 반드시 JSON 형식 유지

기사 원문:
{context}
"""

    try:
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=500
        )
        result = response.choices[0].message.content.strip()
        summary_json = json.loads(result)
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