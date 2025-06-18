from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI()

# KoBART 모델 로딩
model_name = "digit82/kobart-summarization"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 요청 받을 데이터 형식 정의
class SummaryRequest(BaseModel):
    texts: List[str]

@app.post("/summarize")
async def summarize(request: SummaryRequest):
    summaries = []
    for text in request.texts:
        if not text.strip():
            summaries.append("요약 없음")
            continue
        try:
            inputs = tokenizer.encode(text[:1024], return_tensors="pt", truncation=True)
            summary_ids = model.generate(
                inputs,
                max_length=256,
                min_length=20,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)
        except Exception as e:
            summaries.append("요약 실패")
    return {"summaries": summaries}