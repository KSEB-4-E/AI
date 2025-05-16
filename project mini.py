pip install openai
import openai

# OpenAI API 키 입력 (환경변수에 저장하거나 코드에 직접 입력 가능)
openai.api_key = "sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

def summarize_news(article_text):
    prompt = f"""
You are a helpful assistant that summarizes news articles.
Summarize the following article in 3-4 sentences:

\"\"\"{article_text}\"\"\"
"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",   # 또는 gpt-4 사용 가능
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=300
    )

    summary = response.choices[0].message['content'].strip()
    return summary

# 🔍 예시 기사
article = """
The United Nations held an emergency meeting today to address the rising tensions in the Middle East.
Several countries expressed concern about the recent escalation and called for diplomatic solutions.
The Secretary-General emphasized the need for restraint and dialogue between the parties involved.
More discussions are scheduled for later this week to find a peaceful resolution to the conflict.
"""

# 📝 요약 실행
summary_result = summarize_news(article)
print("🧾 요약 결과:\n", summary_result)