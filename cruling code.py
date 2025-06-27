# 코랩 환경에서는 처음 한 번만 설치!
# !pip install feedparser requests pandas beautifulsoup4 transformers sentencepiece

import time
import requests
import pandas as pd
import feedparser
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datetime import datetime

# --- KoBART 모델 로드 (로컬에서 한 번만) ---
model_name = "digit82/kobart-summarization"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def summarize_kobart(text):
    try:
        if not isinstance(text, str) or text.strip() == "":
            return "요약 없음"
        inputs = tokenizer.encode(text[:1024], return_tensors="pt", truncation=True)
        summary_ids = model.generate(inputs, max_length=256, min_length=20, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        print(f"[Error] 요약 실패 - {e}")
        return "요약 실패"

def extract_body(url, source=None):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, timeout=7, headers=headers)
        soup = BeautifulSoup(res.text, "html.parser")
        # --- 신문사별 selector 적용 ---
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

        # --- 불필요한 줄/광고/저작권 안내/기자명 등 정제 ---
        lines = [line.strip() for line in body.splitlines() if len(line.strip()) > 20]
        ban_keywords = [
            "기자", "무단전재", "저작권", "재배포", "모바일로 보기", "무단 복제", "ⓒ", "사진=", "광고", "본문보기", "앱에서 보기"
        ]
        lines = [line for line in lines if not any(bk in line for bk in ban_keywords)]
        clean_body = "\n".join(lines)
        if not clean_body or len(clean_body) < 30:
            return "본문 없음"
        return clean_body
    except Exception as e:
        print(f"[본문 추출 오류]: {e}")
        return "본문 없음"

# --- RSS 피드 목록 ---
rss_feeds = {
    "전자신문": "https://rss.etnews.com/Section901.xml",
    "한겨레": "https://www.hani.co.kr/rss/",
    "SBS": "https://news.sbs.co.kr/news/SectionRssFeed.do?sectionId=01",
    "매일경제": "https://www.mk.co.kr/rss/40300001/",
    "세계일보": "https://www.segye.com/Articles/RSSList/segye_recent.xml"
}

# --- 본문 크롤링 + 요약 + 저장 ---
data = []
for source, rss_url in rss_feeds.items():
    print(f"[INFO] {source} 처리 중...")
    try:
        feed = feedparser.parse(rss_url)
        for entry in feed.entries[:25]:  # 기사 수 조절
            try:
                title = entry.title.strip().replace("\n", " ").replace(",", " ")
                link = entry.link
                content = extract_body(link, source=source)

                if content == "본문 없음":
                    summary = "요약 생략 (본문 부족)"
                else:
                    summary = summarize_kobart(content)
                    if len(summary) > len(content) * 3:
                        summary = f"(비정상 요약 생략) 본문 {len(content)}자, 요약 {len(summary)}자"

                print(f" → {source}: '{title[:40]}...' | 본문 {len(content)}자 | 요약 {len(summary)}자")

                data.append({
                    "source": source,
                    "title": title,
                    "link": link,
                    "content": content,
                    "summary": summary
                })
                time.sleep(0.4)
            except Exception as e:
                print(f"[ERROR] {source} 개별 기사 처리 실패 - {e}")
    except Exception as e:
        print(f"[ERROR] {source} RSS 처리 실패 - {e}")

# --- DataFrame으로 저장 ---
today = datetime.today().strftime("%Y%m%d")
df = pd.DataFrame(data)
df = df.drop_duplicates(subset='title')
df.to_csv(f"kobart_news_{today}.csv", index=False, encoding="utf-8-sig")
print("✅ 모든 뉴스 크롤링 및 저장 완료!")