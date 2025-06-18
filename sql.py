import mysql.connector

# ✅ MySQL 접속 정보
conn = mysql.connector.connect(
    host="127.0.0.1",       # 또는 외부 DB IP
    user="KSEB",            # 사용자 계정
    password="kseb#8981",        # 비밀번호
    database="newsdb"       # DB 이름
)

cursor = conn.cursor()

# ✅ SQL 실행 (뉴스 원문 조회)
cursor.execute("SELECT content FROM news")  # 테이블명/컬럼명은 너희 DB에 맞게
rows = cursor.fetchall()

for row in rows:
    article_id, title, content = row
    print(f"[{content[:100]}...\n")  # 100자 미리보기

cursor.close()
conn.close()