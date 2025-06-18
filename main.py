from fastapi import FastAPI, BackgroundTasks, Query
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
    except:
        return "본문 없음"

def summarize_kobart(text):
    try:
        if not text.strip():
            return "요약 없음"
        inputs = tokenizer.encode(text[:1024], return_tensors="pt", truncation=True)
        summary_ids = model.generate(inputs, max_length=256, min_length=20, length_penalty=2.0, num_beams=4, early_stopping=True)
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except:
        return "요약 실패"

def save_to_sqlite(df, db_path=None, table_name="news"):
    base_dir = os.path.dirname(__file__)
    db_path = os.path.join(base_dir, "news_articles.db")
    today = datetime.today().strftime("%Y%m%d")

    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        for _, row in df.iterrows():
            cur.execute(f"""
                INSERT INTO {table_name} (source, title, link, content, summary, date)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (row['source'], row['title'], row['link'], row['content'], row['summary'], today))
        conn.commit()
        conn.close()
        print("[DB 저장 완료 ✅]")
    except Exception as e:
        print(f"[DB 저장 실패 ❌]: {e}")

def run_news_job():
    try:
        print(f"[{datetime.now()}] 뉴스 수집 시작")
        data = []
        for source, rss_url in rss_feeds.items():
            feed = feedparser.parse(rss_url)
            for entry in feed.entries[:25]:
                title = entry.title.strip().replace("\n", " ").replace(",", " ")
                link = entry.link
                content = extract_body(link)
                summary = "요약 생략 (본문 부족)" if content == "본문 없음" else summarize_kobart(content)
                data.append({
                    "source": source, "title": title, "link": link,
                    "content": content, "summary": summary
                })
                time.sleep(0.2)
        df = pd.DataFrame(data).drop_duplicates(subset="title")
        save_to_sqlite(df)
        print(f"[{datetime.now()}] ✅ 뉴스 저장 완료")
    except Exception as e:
        print(f"[뉴스 수집 실패 ❌]: {e}")

def extract_keywords(texts, top_n=5):
    stopwords = set(["그리고", "그러나", "하지만", "또한", "등", "이", "그", "저", "것", "수", "으로", "들", "에서", "하다", "한", "대해",
        "있다", "검색해줘", "세계로", "이끌어줄", "대한","밝혔다", "후보는", "칠한", "지난", "있는", "주요", "로", "은", "는", "이", "가",
        "을", "를", "에", "의", "와", "과", "도", "것으로", "가운데", "대통령은", "나눔의", "대통령이", "물론", "되겠다",
        "만에", "내일", "당신의", "기사를", "동향과", "정부의","바탕으로", "위한", "위해", "정부는", "장거리요","개최했다","최근","휘두른","마치고"
        ,"빠른","통해","무슨","있으며","기술을","탐험과","시장의","숨져","네이버의","명의","것이","지원을","커지고","기업의","씨가","문제를","이후",
        "타더니","전망도","등을","받고","기업들의","없는","가능성까지","개에","다시","검색해줘최근","마리","모두","있다고","알림을","함께","내용을",
        "개방하는","우리가","열린","것을","관련","등록일자","라는","관심","추가", "관심", "활용해","지적","따르면","강한","마친","나오고","방안을",
        "중요하다고","의지를","구체적인","논란뷰티당신의","기사","씨를","기반","맞아","택시에서","크게","강력","사례가","매경","맥락을","발표된","위를",
        "각각","최신","이를","데이터를","국민에게","발견","상승률은"
                     ])
    try:
        tokenized = []
        for text in texts:
            words = re.findall(r"[가-힣]{2,}", text)
            words = [w for w in words if w not in stopwords]
            tokenized.append(" ".join(words))
        if not any(tokenized):
            return []
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(tokenized)
        feature_names = vectorizer.get_feature_names_out()
        word_counts = (X > 0).sum(axis=0).A1
        keywords = [(feature_names[i], int(word_counts[i])) for i in range(len(feature_names))]
        keywords.sort(key=lambda x: x[1], reverse=True)
        return [{"keyword": w, "count": c} for w, c in keywords[:top_n]]
    except:
        return []

# ===================== [FastAPI 엔드포인트] =====================
@app.get("/run-news")
def trigger_run(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_news_job)
    return {"message": "뉴스 수집을 시작했습니다."}

@app.get("/trending-keywords")
def get_trending_keywords():
    try:
        db_path = os.path.join(os.path.dirname(__file__), "news_articles.db")
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT title, summary FROM news", conn)
        conn.close()
        combined = (df["title"].fillna("") + " " + df["summary"].fillna(""))
        sampled = random.sample(combined.tolist(), min(40, len(combined)))
        keywords = extract_keywords(sampled, top_n=5)
        return {"keywords": keywords}
    except Exception as e:
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
        return {
            "keyword": keyword,
            "summary": {"fact": "요약 실패", "issue": "요약 실패", "outlook": "요약 실패"},
            "error": str(e)
        }

# ===================== [스케줄러 등록] =====================
@app.on_event("startup")
def start_scheduler():
    ensure_db()
    scheduler = BackgroundScheduler()
    scheduler.add_job(run_news_job, "interval", hours=1)
    scheduler.start()
    print("⏰ 스케줄러 시작됨: 1시간마다 뉴스 수집")

from fastapi.responses import JSONResponse

@app.get("/debug-news")
def debug_news():
    try:
        db_path = os.path.join(os.path.dirname(__file__), "news_articles.db")
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT source, title, summary, date FROM news ORDER BY date DESC LIMIT 5", conn)
        conn.close()
        return JSONResponse(content=df.to_dict(orient="records"))
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    run_news_job()