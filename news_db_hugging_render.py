from fastapi import FastAPI, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
import pandas as pd
import feedparser
import requests
import sqlite3
import time
import os
import json
from bs4 import BeautifulSoup
from datetime import datetime
from fastapi.responses import JSONResponse
from kiwipiepy import Kiwi
import threading
import random

# ===================== [초기 설정] =====================
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
if not HF_API_TOKEN:
    raise RuntimeError("❌ Hugging Face API 토큰이 설정되지 않았습니다. .env 파일 확인 요망.")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

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

# ===================== [본문 크롤링 (selector 적용)] =====================
def extract_body(url, source=None):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, timeout=7, headers=headers)
        soup = BeautifulSoup(res.text, "html.parser")

        # 신문사별 selector
        if source == "한겨레":
            article = soup.select_one("div.article-text")
        elif source == "전자신문":
            article = soup.select_one("#articleBody")
        elif source == "SBS":
            article = soup.select_one("div.text_area")
        elif source == "매일경제":
            article = soup.select_one("div#article_body")
        elif source == "세계일보":
            article = soup.select_one("div#article_txt")
        else:
            article = None

        if article:
            body = article.get_text(separator="\n").strip()
        else:
            paragraphs = soup.find_all("p")
            body = "\n".join(p.get_text().strip() for p in paragraphs if p.get_text().strip())

        lines = [line.strip() for line in body.splitlines() if len(line.strip()) > 20]
        lines = [line for line in lines if "기자" not in line and "무단전재" not in line]
        clean_body = "\n".join(lines)
        if not clean_body or len(clean_body) < 30:
            return "본문 없음"
        return clean_body
    except Exception as e:
        print(f"[본문 추출 오류]: {e}")
        return "본문 없음"

# ===================== [KoBART 요약 - HF API] =====================
def summarize_kobart(text):
    for attempt in range(2):  # 최대 2회 재시도
        try:
            print("✏️ KoBART 요약 요청")
            api_url = "https://api-inference.huggingface.co/models/digit82/kobart-summarization"
            headers = {
                "Authorization": f"Bearer {HF_API_TOKEN}",
                "Content-Type": "application/json"
            }

            payload = {
                "inputs": text[:1024],
                "options": {"wait_for_model": True}
            }

            response = requests.post(api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            if isinstance(result, list) and "summary_text" in result[0]:
                return result[0]["summary_text"]
            return "요약 실패"
        except Exception as e:
            print(f"⚠️ 요약 재시도 {attempt + 1}회 실패: {e}")
            time.sleep(1)
    return "요약 실패"

# ===================== [DB 저장] =====================
def save_to_sqlite(df, db_path=None, table_name="news", max_articles=150):
    base_dir = os.path.dirname(__file__)
    db_path = os.path.join(base_dir, "news_articles.db")
    print(f"[DB 저장 경로]: {db_path}")
    today = datetime.today().strftime("%Y%m%d")
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
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

# ===================== [중복 실행 방지 락] =====================
run_lock = threading.Lock()

def run_news_job():
    if not run_lock.acquire(blocking=False):
        print("⚠️ 이미 뉴스 수집이 진행중입니다. 중복 실행 방지.")
        return
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
                content = extract_body(link, source=source)
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
    finally:
        run_lock.release()

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
        "로", "으로", "에", "의", "와", "과", "도", "것으로", "가운데", "대통령은", "나눔의", "대통령이", "물론", "되겠다", "업무", "보고"
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
def run_news_direct(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_news_job)
    return {"message": "뉴스 수집을 백그라운드에서 시작했습니다."}

@app.get("/trending-keywords")
def get_trending_keywords():
    try:
        db_path = os.path.join(os.path.dirname(__file__), "news_articles.db")
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT title, summary FROM news", conn)
        conn.close()
        recent = df.tail(150)
        if len(recent) < 50:
            sampled = recent
        else:
            sampled = recent.sample(n=50, random_state=None)
        combined = (sampled["title"].fillna("") + " " + sampled["summary"].fillna(""))
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
    당신은 다수의 뉴스 기사를 분석해 본질적 쟁점을 도출하는 뉴스 요약 전문가입니다.

    아래는 '{keyword}'와 관련된 여러 언론사의 기사 원문입니다.
    기사의 중복 표현이나 단순 사실 나열이 아닌, **핵심만 간결하게** 정리해 주세요.

    다음 3가지 항목을 기준으로, **각 항목을 반드시 1문장**으로 정리해 주세요.

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