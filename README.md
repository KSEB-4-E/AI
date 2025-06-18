# 🟡 핵심: 이게 없으면 프론트 요청은 무조건 404남
@app.get("/api/search")
def search(q: Optional[str] = Query(None)):
    print("검색어:", q)  # 콘솔 확인용 로그
    if not q:
        return {"results": [], "message": "검색어가 없습니다."}

    # 👉 나중에 DB 연동하거나 검색 알고리즘 넣으면 됨
    return {
        "results": [f"🔍 '{q}'에 대한 가짜 검색 결과입니다.", "예시 1", "예시 2"],
        "count": 3
    }

