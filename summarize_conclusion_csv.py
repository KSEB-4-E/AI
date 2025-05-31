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

# uvicorn summarize_conclusion_csv:app --reload
# 키워드: http://localhost:8000/trending-keywords
#
# 본문검색: http://localhost:8000/search-articles?keyword=카카오
#
# http://localhost:8000/docs

# ✅ 환경 변수 로딩 및 GPT 초기화
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ CSV 파일 미리 로드
df = pd.read_csv("kobart_news_summarized.csv", encoding="cp949")

# ✅ FastAPI 앱 초기화
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 서버 시작 시점에 키워드 추천 미리 캐싱
kiwi = Kiwi()
cached_keywords = []

def extract_trending_keywords():
    texts = (df["title"].fillna("") + " " + df["summary"].fillna("")).tolist()[:100]
    all_keywords = []
    for text in texts:
        all_keywords += [
            token.form for token in kiwi.tokenize(text)
            if token.tag in ["NNG", "NNP"] and len(token.form) > 1
        ]
    most_common = Counter(all_keywords).most_common(5)
    return [{"keyword": kw, "count": count} for kw, count in most_common]

cached_keywords = extract_trending_keywords()

# ✅ 1. 추천 키워드 API
@app.get("/trending-keywords")
def get_trending_keywords():
    return {"keywords": cached_keywords}

# ✅ 2. 기사 검색 API
@app.get("/search-articles")
def search_articles(keyword: str = Query(..., min_length=2)):
    # 1. 키워드 기반 필터링 (title/summary에서 검색)
    filtered = df[
        df["title"].fillna("").str.contains(keyword, case=False, regex=False) |
        df["summary"].fillna("").str.contains(keyword, case=False, regex=False)
    ].copy()

    # 2. 최신 기사 우선 → 전체 데이터에서 아래쪽이 최신이므로 reverse index 부여
    filtered["row_order"] = filtered.index[::-1]

    # 3. 신문사(source)별로 최신 기사 하나씩 추출
    latest_by_source = (
        filtered.sort_values("row_order")  # 아래에 있는 기사가 최신
        .drop_duplicates(subset=["source"], keep="first")  # 신문사별 첫 번째(최신) 기사
    )

    # 4. 가장 최신 기사 3개만 추출
    latest_articles = latest_by_source.sort_values("row_order").head(3)

    # 5. 필요한 컬럼만 포함
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

    각 기사에서 공통적으로 다루는 이슈의 핵심을 다음 세 항목으로 나누어 요약해줘:

    1. 핵심 사실 (무슨 일이 일어났는가?)  
    2. 언론의 전반적인 입장 요약 (언론들은 어떻게 해석하고 있는가?)  
    3. 향후 전망 또는 종합 판단 (앞으로 어떻게 될 가능성이 있는가?)  

    - 각 항목은 한 문장 이내로 간결하고 명확하게 서술해줘.
    - 직접적인 인용보다는 핵심 개념을 요약해서 전달해줘.
    - 중립적이고 객관적인 표현을 사용해줘.

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