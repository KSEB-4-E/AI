from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from apscheduler.schedulers.background import BackgroundScheduler
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv
import pandas as pd
import feedparser
import requests
import sqlite3
import time
import os
import json
import random
import re
from bs4 import BeautifulSoup
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import openai
from fastapi.responses import JSONResponse
from kiwipiepy import Kiwi

# ===================== [초기 설정] =====================
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

model_name = "digit82/kobart-summarization"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

rss_feeds = {
    "전자신문": "https://rss.etnews.com/Section901.xml",
    "한겨레": "https://www.hani.co.kr/rss/",
    "SBS": "https://news.sbs.co.kr/news/SectionRssFeed.do?sectionId=01",
    "매일경제": "https://www.mk.co.kr/rss/40300001/",
    "세계일보": "https://www.segye.com/Articles/RSSList/segye_recent.xml"
}

kiwi = Kiwi()

# ===================== [DB 보장 함수] =====================
def ensure_db():
    db_path = os.path.join(os.path.dirname(__file__), "news_articles.db")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS news (
            source TEXT,
            title TEXT,
            link TEXT,
            content TEXT,
            summary TEXT,
            date TEXT
        )
    """)
    conn.commit()
    conn.close()
    print("✅ news 테이블 확인 또는 생성 완료")

# ===================== [핵심 기능] =====================
def extract_body(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, timeout=5, headers=headers)
        soup = BeautifulSoup(res.text, "html.parser")
        paragraphs = soup.find_all("p")
        body = "\n".join(p.get_text().strip() for p in paragraphs if p.get_text().strip())
        if not body or len(body) < 30 or any(kw in body.lower() for kw in ["삭제", "없음", "404"]):
            return "본문 없음"
        return body
    except Exception as e:
        print(f"[본문 추출 오류]: {e}")
        return "본문 없음"

def summarize_kobart(text):
    try:
        print("✏️ KoBART 요약 시작")
        if not text.strip():
            return "요약 없음"
        inputs = tokenizer.encode(text[:1024], return_tensors="pt", truncation=True)
        summary_ids = model.generate(inputs, max_length=256, min_length=20, length_penalty=2.0, num_beams=4, early_stopping=True)
        result = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        print("✅ 요약 완료")
        return result
    except Exception as e:
        print(f"❌ 요약 실패: {e}")
        return "요약 실패"

def save_to_sqlite(df, db_path=None, table_name="news", max_articles=150):
    base_dir = os.path.dirname(__file__)
    db_path = os.path.join(base_dir, "news_articles.db")
    print(f"[DB 저장 경로]: {db_path}")
    today = datetime.today().strftime("%Y%m%d")
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        # 현재 저장된 뉴스 개수
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        current_count = cur.fetchone()[0]
        new_count = len(df)
        delete_count = max(0, (current_count + new_count) - max_articles)
        if delete_count > 0:
            cur.execute(f"DELETE FROM {table_name} WHERE rowid IN (SELECT rowid FROM {table_name} ORDER BY date, rowid LIMIT ?)", (delete_count,))
            print(f"🗑️ {delete_count}개 오래된 기사 삭제")
        for _, row in df.iterrows():
            try:
                cur.execute(f"""
                    INSERT INTO {table_name} (source, title, link, content, summary, date)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (row['source'], row['title'], row['link'], row['content'], row['summary'], today))
            except Exception as row_e:
                print(f"❌ 개별 저장 실패: {row['title']} | 이유: {row_e}")
        conn.commit()
        conn.close()
        print("[DB 저장 완료 ✅]")
    except Exception as e:
        print(f"[DB 저장 실패 ❌]: {e}")

def run_news_job():
    try:
        print(f"\n[{datetime.now()}] 📰 뉴스 수집 시작")
        data = []
        per_feed = max(1, 150 // len(rss_feeds))
        for source, rss_url in rss_feeds.items():
            print(f"📡 [RSS 요청] 언론사: {source} | URL: {rss_url}")
            feed = feedparser.parse(rss_url)
            print(f"✅ [RSS 수신 완료] {len(feed.entries)}개 기사 발견")
            for entry in feed.entries[:per_feed]:
                title = entry.title.strip().replace("\n", " ").replace(",", " ")
                link = entry.link
                print(f"🔗 기사 제목: {title}")
                print(f"🧭 기사 링크: {link}")
                content = extract_body(link)
                print(f"📄 본문 길이: {len(content)}")
                if content == "본문 없음":
                    print("⚠️ 본문 없음 - 요약 생략")
                    summary = "요약 생략 (본문 부족)"
                else:
                    summary = summarize_kobart(content)
                    print(f"📚 요약 내용: {summary[:50]}...")
                data.append({
                    "source": source,
                    "title": title,
                    "link": link,
                    "content": content,
                    "summary": summary
                })
                time.sleep(0.1)
        df = pd.DataFrame(data).drop_duplicates(subset="title")
        print(f"💾 최종 저장 대상: {len(df)}건 / 원본: {len(data)}건")
        if df.empty:
            print("❌ 저장할 데이터 없음")
        else:
            print(f"✅ DB 저장")
            save_to_sqlite(df)
        print(f"[{datetime.now()}] ✅ 뉴스 저장 완료")
    except Exception as e:
        print(f"[🔥 예외 발생] 뉴스 수집 실패: {e}")

# === 키위 기반 명사 추출 & 키워드 함수 ===
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
        "로", "으로", "에", "의", "와", "과", "도", "것으로", "가운데", "대통령은", "나눔의", "대통령이", "물론", "되겠다",
        # 추가 필요시 계속 보며 관리!
    ])
    all_nouns = []
    for text in texts:
        nouns = extract_nouns_kiwi(text)
        all_nouns.extend([n for n in nouns if n not in stopwords and len(n) > 1])
    from collections import Counter
    counter = Counter(all_nouns)
    return [{"keyword": w, "count": c} for w, c in counter.most_common(top_n)]

# ===================== [FastAPI 엔드포인트] =====================
@app.get("/run-news")
def run_news_direct():
    try:
        run_news_job()
        return {"message": "뉴스 수집을 즉시 완료했습니다."}
    except Exception as e:
        return {"error": str(e)}

@app.get("/trending-keywords")
def get_trending_keywords():
    try:
        db_path = os.path.join(os.path.dirname(__file__), "news_articles.db")
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT title, summary FROM news", conn)
        conn.close()
        recent = df.tail(150)
        sampled = random.sample(recent.index.tolist(), min(30, len(recent)))
        combined = (df.loc[sampled, "title"].fillna("") + " " + df.loc[sampled, "summary"].fillna(""))
        keywords = extract_keywords_kiwi(combined.tolist(), top_n=5)
        return {"keywords": keywords}
    except Exception as e:
        print(f"❌ trending-keywords 오류: {e}")
        return {"error": str(e)}

@app.get("/search-articles")
def search_articles(keyword: str = Query(..., min_length=2)):
    try:
        db_path = os.path.join(os.path.dirname(__file__), "news_articles.db")
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT source, title, summary, content, link FROM news", conn)
        conn.close()
        df = df.fillna("")
        filtered = df[
            df["title"].str.contains(keyword, case=False, na=False) |
            df["summary"].str.contains(keyword, case=False, na=False)
        ].copy()
        filtered["row_order"] = filtered.index[::-1]
        latest_by_source = filtered.sort_values("row_order").drop_duplicates("source")
        articles = latest_by_source.sort_values("row_order").head(3)
        return {
            "keyword": keyword,
            "articles": articles[["title", "summary", "content", "source", "link"]].to_dict(orient="records")
        }
    except Exception as e:
        print(f"❌ search-articles 오류: {e}")
        return {"error": str(e)}

class SummaryRequest(BaseModel):
    keyword: str
    contents: List[str]

@app.post("/summarize-conclusion")
def summarize_conclusion(data: SummaryRequest):
    keyword = data.keyword
    context = "\n".join(data.contents[:3])
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
        return {"keyword": keyword, "summary": json.loads(result)}
    except Exception as e:
        print(f"❌ summarize-conclusion 오류: {e}")
        return {
            "keyword": keyword,
            "summary": {"fact": "요약 실패", "issue": "요약 실패", "outlook": "요약 실패"},
            "error": str(e)
        }

@app.get("/debug-news")
def debug_news():
    try:
        db_path = os.path.join(os.path.dirname(__file__), "news_articles.db")
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT source, title, summary, date FROM news ORDER BY date DESC LIMIT 10", conn)
        conn.close()
        return JSONResponse(content=df.to_dict(orient="records"))
    except Exception as e:
        print(f"❌ debug-news 오류: {e}")
        return {"error": str(e)}

# ===================== [스케줄러 등록] =====================
@app.on_event("startup")
def start_scheduler():
    ensure_db()
    scheduler = BackgroundScheduler()
    scheduler.add_job(run_news_job, "interval", hours=1)
    scheduler.start()
    print("⏰ 스케줄러 시작됨: 1시간마다 뉴스 수집")

if __name__ == "__main__":
    run_news_job()