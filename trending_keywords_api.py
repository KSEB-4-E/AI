import pandas as pd
from kiwipiepy import Kiwi
from collections import Counter
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ✅ 1. CSV 파일 불러오기
csv_path = "kobart_news_summarized.csv"  # 같은 디렉토리에 위치
df = pd.read_csv(csv_path, encoding="cp949")

# ✅ 2. 최근 100개의 title + summary 텍스트 결합
texts = (df["title"].fillna("") + " " + df["summary"].fillna("")).tolist()[:100]

# ✅ 3. 키워드 추출 함수 (명사 기반, 길이 2 이상)
def extract_keywords(texts, top_n=10):
    kiwi = Kiwi()
    all_keywords = []
    for text in texts:
        tokens = kiwi.tokenize(text)
        keywords = [
            token.form for token in tokens
            if token.tag in ['NNG', 'NNP'] and len(token.form) > 1
        ]
        all_keywords.extend(keywords)
    counter = Counter(all_keywords)
    return [kw for kw, _ in counter.most_common(top_n)]

# ✅ 4. FastAPI 앱 초기화
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프론트 도메인 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 5. 추천 검색어 API 엔드포인트
@app.get("/trending-keywords")
def get_trending_keywords():
    keywords = extract_keywords(texts)
    return {"keywords": keywords}