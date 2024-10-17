from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import fasttext
import os
from bpe_tokenizer_for_model import bpe_tokenize, train_tokenizer, load_tokenizer
from fastapi.staticfiles import StaticFiles


app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")


# 모델 경로 및 데이터 경로 설정
model_path = r'C:\Users\User\Desktop\PythonWorkspace\item_category_recommend\scripts\project\model_result\tokenized_product_category_from_fullset_model_epoch15.bin'
data_path = r'C:\Users\User\Desktop\PythonWorkspace\item_category_recommend\train_test_data\full_output\before_processed\for_bpe'
tokenizer_path = r'C:\Users\User\Desktop\PythonWorkspace\item_category_recommend\scripts\project\tokenizer.json'

# FastText 모델 로드 또는 학습
if os.path.exists(model_path):
    model = fasttext.load_model(model_path)
    print("Prediction Model successfully loaded.")
else:
    # 모델이 없을 경우 처리 (예: 오류 메시지 출력)
    print("Model not found. Please train the model first.")
    exit()

# 토크나이저 로드 또는 학습
if os.path.exists(tokenizer_path):
    load_tokenizer()
    print("BPE Tokenizer successfully loaded.")
else:
    train_tokenizer(data_path)
    print("BPE Tokenizer successfully trained")

# 메인 페이지 엔드포인트
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 추천 API 엔드포인트
@app.get("/recommend")
async def recommend(query: str):
    # 입력 쿼리를 토크나이즈
    tokenized_query = bpe_tokenize(query)
    
    # 모델 예측 수행
    result = model.predict(tokenized_query, k=3)
    
    # 결과가 있을 경우
    if result:
        labels = [label.replace('__label__', '') for label in result[0]]
        probabilities = result[1]
        recommendations = [{"label": label, "probability": prob} for label, prob in zip(labels, probabilities)]
    else:
        recommendations = []
    
    return {"recommendations": recommendations}
