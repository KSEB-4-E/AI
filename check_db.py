import sqlite3
import pandas as pd
import os

# DB 경로 설정
db_path = os.path.join(os.path.dirname(__file__), "news_articles.db")

# DB 연결
try:
    conn = sqlite3.connect(db_path)
    print(f"✅ DB 연결 성공: {db_path}")

    # 테이블 존재 확인
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    print(f"📌 테이블 목록: {[t[0] for t in tables]}")

    # 뉴스 테이블이 있다면 상위 5개 출력
    if ('news',) in tables:
        df = pd.read_sql_query("SELECT summary, date FROM news ORDER BY date DESC LIMIT 5", conn)
        print("📰 최근 저장된 뉴스 5건:")
        print(df)
    else:
        print("⚠️ 'news' 테이블이 존재하지 않습니다.")

    conn.close()
except Exception as e:
    print(f"❌ DB 접근 실패: {e}")