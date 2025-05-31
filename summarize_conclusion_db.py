from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
import mysql.connector
import pandas as pd
from kiwipiepy import Kiwi
from collections import Counter
import os
from typing import Optional

# ✅ 환경 변수 로딩
load_dotenv()

# ✅ GPT 클라이언트 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ DB에서 뉴스 데이터 불러오기 함수
def fetch_news_from_db():
    conn = mysql.connector.connect(
        host=os.getenv("DB_HOST"),       # 예: '127.0.0.1'
        user=os.getenv("DB_USER"),       # 예: 'root'
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME")    # 예: 'newsdb'
    )
    cursor = conn.cursor()
    cursor.execute("SELECT title, summary, content FROM news")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return pd.DataFrame(rows, columns=["title", "summary", "content"])

# ✅ FastAPI 앱 초기화
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 전역 데이터프레임 불러오기
df = fetch_news_from_db()

@app.get("/api/search")
def search(q: Optional[str] = Query(None)):
    print("검색어:", q) # 콘솔 확인용 로그
    if not q:
        return {"results": [], "message": "검색어가 없습니다."}

# 👉 나중에 DB 연동하거나 검색 알고리즘 넣으면 됨
    return {
        "results": [f"🔍 '{q}'에 대한 가짜 검색 결과입니다.", "예시 1", "예시 2"],
        "count": 3
    }

# ✅ 추천 키워드 API
@app.get("/trending-keywords")
def get_trending_keywords():
    texts = (df["title"].fillna("") + " " + df["summary"].fillna("")).tolist()[:100]
    kiwi = Kiwi()
    all_keywords = []
    for text in texts:
        all_keywords += [token.form for token in kiwi.tokenize(text)
                         if token.tag in ["NNG", "NNP"] and len(token.form) > 1]
    most_common = Counter(all_keywords).most_common(10)
    return {"keywords": [kw for kw, _ in most_common]}

# ✅ 기사 검색 API
@app.get("/search-articles")
def search_articles(keyword: str = Query(..., min_length=2)):
    filtered = df[
        df["title"].fillna("").str.contains(keyword, case=False, regex=False) |
        df["summary"].fillna("").str.contains(keyword, case=False, regex=False) |
        df["content"].fillna("").str.contains(keyword, case=False, regex=False)
    ]
    articles = filtered[["title", "summary", "content"]].dropna().head(5).to_dict(orient="records")
    return {"keyword": keyword, "articles": articles}

# ✅ 결론 요약 요청 모델
class SummaryRequest(BaseModel):
    keyword: str
    contents: List[str]

# ✅ 결론 요약 API
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