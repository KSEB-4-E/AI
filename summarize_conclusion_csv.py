from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from kiwipiepy import Kiwi
import pandas as pd
import os
import json
import random
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

kiwi = Kiwi()

def extract_nouns_kiwi(text):
    nouns = []
    for word, pos, _, _ in kiwi.analyze(text)[0][0]:
        if pos in ("NNG", "NNP"):
            nouns.append(word)
    return nouns

def extract_keywords_kiwi(texts, top_n=5):
    stopwords = set([
        "뉴스", "기자", "한국", "정부", "오늘", "제공", "관련", "보도", "사실", "통해", "위해",
        "등", "이", "그", "저", "것", "수", "명", "제", "시", "때", "후", "위", "앞", "뒤",
        "중", "내", "밖", "이후", "위해", "대해", "대한", "에", "와", "과", "는", "이", "가", "을", "를",
        "로", "으로", "에", "의", "와", "과", "도", "것으로", "가운데", "대통령은", "나눔의", "대통령이", "물론", "되겠다"
    ])
    all_nouns = []
    for text in texts:
        nouns = extract_nouns_kiwi(text)
        all_nouns.extend([n for n in nouns if n not in stopwords and len(n) > 1])
    from collections import Counter
    counter = Counter(all_nouns)
    return [{"keyword": w, "count": c} for w, c in counter.most_common(top_n)]

@app.get("/trending-keywords")
def get_trending_keywords():
    try:
        df = pd.read_excel("kobart_news_summarized.xlsx")
        if len(df) < 50:
            sampled = df["title"].fillna("").tolist()  # 기사 50개 미만이면 전부 사용
        else:
            sampled_idx = random.sample(list(df.index), 50)
            sampled = df.loc[sampled_idx, "title"].fillna("").tolist()
        keywords = extract_keywords_kiwi(sampled, top_n=5)
        return {"keywords": keywords}
    except Exception as e:
        return {"error": str(e)}

@app.get("/search-articles")
def search_articles(keyword: str = Query(..., min_length=2)):
    try:
        df = pd.read_excel("kobart_news_summarized.xlsx")
        filtered = df[
            df["title"].fillna("").str.contains(keyword, case=False, regex=False)
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
    context = "\n".join(data.contents[:3])
    prompt = f"""
당신은 다수의 뉴스 기사를 분석해 본질적 쟁점을 도출하는 뉴스 요약 전문가입니다.

아래는 '{keyword}'와 관련된 여러 언론사의 기사 원문입니다.
중복 표현/사실 나열 없이, **핵심만 간결하게** 정리해 주세요.

아래 3가지 항목을 기준으로, **각 항목을 반드시 1문장**으로 정리해 주세요.

1. "fact":  
    - 여러 기사에 반복적으로 등장하는 **가장 중요한 핵심 사실**을 명확하게,  
    - **팩트(객관적 진술)**만 담아 요약

2. "issue":  
    - 해당 뉴스 이슈의 **주요 쟁점/갈등/논란/사회적 반향**을 요약  
    - 언론사들의 공통적으로 주목한 **문제점이나 논쟁**을 한 문장으로 명확하게 서술

3. "outlook":  
    - 전문가 또는 언론들이 제시한 **미래 전망, 영향, 시사점**을  
    - **비판적/통합적 관점**에서 한 문장으로 정리    

**형식**
- 아래 JSON 예시처럼, 각 항목 이름은 "fact", "issue", "outlook" 그대로, 반드시 JSON 형식으로만 답변
- 직접 인용 없이 당신의 언어로 요약, 절대 기사 문장 그대로 복사 금지
- 모호하거나 과장된 표현, 추측은 지양
- 예측/전망(outlook)은 기사 내에 근거가 있을 때만 제시

예시:
{{
  "fact": "...",
  "issue": "...",
  "outlook": "..."
}}

[기사 원문 모음]
{context}
""".strip()

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