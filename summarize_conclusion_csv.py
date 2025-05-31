from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
from kiwipiepy import Kiwi
from collections import Counter
import os

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

# ✅ 1. 추천 키워드 API (요청 시 처리, 30개 기사 기준)
@app.get("/trending-keywords")
def get_trending_keywords():
    df = pd.read_csv("kobart_news_summarized.csv", encoding="cp949")
    texts = (df["title"].fillna("") + " " + df["summary"].fillna("")).tolist()[:30]
    kiwi = Kiwi()
    all_keywords = []
    for text in texts:
        all_keywords += [
            token.form for token in kiwi.tokenize(text)
            if token.tag in ["NNG", "NNP"] and len(token.form) > 1
        ]
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

    return {
        "keyword": keyword,
        "articles": articles
    }

# ✅ 3. 결론 요약 API
class SummaryRequest(BaseModel):
    keyword: str
    contents: List[str]

@app.post("/summarize-conclusion")
def summarize_conclusion(data: SummaryRequest):
    keyword = data.keyword
    contents = data.contents[:3]
    prompt = f"""
    다음은 '{keyword}'에 대한 여러 언론사의 기사 원문입니다.

    다음 세 항목에 따라 전체 이슈를 요약해줘:

    1. 핵심 사실  
       - 무슨 일이 일어났는지 요약해줘

    2. 각 신문사 기사들의 공통된 쟁점  
       - 기사들이 공통으로 다루는 주요 쟁점을 정리해줘

    3. 향후 전망 또는 종합 판단  
       - 앞으로 벌어질 가능성이 있는 일이나 전체적인 해석을 제시해줘

    ※ 조건:
    - 각 항목은 줄을 바꿔 **하나의 항목당 한 문장 이내**로 간결하게 써줘
    - 직접 인용 대신 핵심 개념을 요약해줘
    - 중립적이고 객관적인 표현으로 작성해줘

    기사 원문:
    """ + "\n".join(contents)
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