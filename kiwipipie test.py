from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
import random
import json

from kiwipiepy import Kiwi
from sklearn.feature_extraction.text import TfidfVectorizer

kiwi = Kiwi()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# 불용어 (명사인데도 자주 나오는 의미 없는 것 위주)
stopwords = set([
    "뉴스", "기자", "한국", "정부", "대한", "대한민국", "오늘", "제공", "기록", "관련", "보도", "사실",
    "통해", "위해", "등", "이", "그", "저", "것", "수", "명", "명으로", "들", "에서", "하다",
    "한", "대해", "있다", "밝혔다", "후보", "지난", "있는", "주요", "로", "은", "는", "이", "가",
    "을", "를", "에", "의", "와", "과", "도", "가운데", "대통령", "물론", "되겠다", "만에", "내일",
    "기사", "내용", "중", "앞", "후", "또", "때", "더", "년", "월", "일", "시간", "최근", "현재"
])

def kiwi_noun_extractor(text):
    """kiwipiepy로 명사만 뽑아 리턴"""
    try:
        return [word for word, tag, _, _ in kiwi.analyze(text)[0][0] if tag.startswith('NN') and word not in stopwords and len(word) > 1]
    except Exception as e:
        print("kiwi 분석 오류:", e)
        return []

def extract_keywords_kiwi(texts, top_n=5):
    try:
        docs = [" ".join(kiwi_noun_extractor(t)) for t in texts]
        docs = [doc for doc in docs if doc.strip()]
        if not docs:
            return []
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(docs)
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = X.sum(axis=0).A1
        word_counts = (X > 0).sum(axis=0).A1

        keywords = []
        for idx, word in enumerate(feature_names):
            if word not in stopwords:
                keywords.append((word, int(word_counts[idx]), tfidf_scores[idx]))
        # count -> tfidf 점수 순(동점시 count) 정렬
        keywords.sort(key=lambda x: (-x[1], -x[2]))
        return [{"keyword": kw, "count": count} for kw, count, _ in keywords[:top_n]]
    except Exception as e:
        print("키워드 추출 오류:", e)
        return []

@app.get("/trending-keywords")
def get_trending_keywords():
    try:
        df = pd.read_excel("kobart_news_summarized.xlsx")
        combined = (df["title"].fillna("") + " " + df["summary"].fillna(""))
        sampled_texts = random.sample(combined.tolist(), min(30, len(combined)))
        keywords = extract_keywords_kiwi(sampled_texts, top_n=5)
        return {"keywords": keywords}
    except Exception as e:
        return {"error": str(e)}

# 명사만 잘 나오는지 테스트하려면 아래처럼 직접 돌려볼 수도 있음
if __name__ == "__main__":
    df = pd.read_excel("kobart_news_summarized.xlsx")
    combined = (df["title"].fillna("") + " " + df["summary"].fillna(""))
    sampled_texts = random.sample(combined.tolist(), min(30, len(combined)))
    print(extract_keywords_kiwi(sampled_texts, top_n=10))