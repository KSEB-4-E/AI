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


# uvicorn summarize_conclusion:app --reload

# http://localhost:8000/trending-keywords
#
# http://localhost:8000/search-articles?keyword=카카오
#
# http://localhost:8000/docs

# ✅ 환경 변수 로딩 및 GPT 클라이언트 초기화
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ CSV 불러오기 (최신 추천 키워드 및 기사 검색용)
df = pd.read_csv("kobart_news_summarized.csv", encoding="cp949")

# ✅ FastAPI 서버 초기화
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프론트 연결 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ✅ 1. 추천 키워드 API
@app.get("/trending-keywords")
def get_trending_keywords():
    texts = (df["title"].fillna("") + " " + df["summary"].fillna("")).tolist()[:100]
    kiwi = Kiwi()
    all_keywords = []
    for text in texts:
        all_keywords += [token.form for token in kiwi.tokenize(text) if
                         token.tag in ["NNG", "NNP"] and len(token.form) > 1]
    most_common = Counter(all_keywords).most_common(10)
    return {"keywords": [kw for kw, _ in most_common]}


# ✅ 2. 기사 검색 API
@app.get("/search-articles")
def search_articles(keyword: str = Query(..., min_length=2)):
    filtered = df[
        df["title"].fillna("").str.contains(keyword, case=False, regex=False) |
        df["summary"].fillna("").str.contains(keyword, case=False, regex=False) |
        df["content"].fillna("").str.contains(keyword, case=False, regex=False)
        ]
    articles = filtered[["title", "summary", "content"]].dropna().head(5).to_dict(orient="records")
    return {"keyword": keyword, "articles": articles}


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

이 기사들의 핵심 내용을 바탕으로 전체 이슈의 흐름을 3문장 이내로 요약해줘.

- 요약은 전체 기사 내용의 약 5~10% 정도 압축된 분량이길 바라.
- 각각의 문장은 (1) 핵심 사실, (2) 언론 입장 요약, (3) 향후 전망 또는 종합 판단 을 담되, 중립적인 시선으로 재구성해줘.
- 직접적인 인용보다는 개념을 정리해서 서술해줘.

""" + "\n\n".join([f"기사 {i + 1}:\n{c}" for i, c in enumerate(contents)])

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