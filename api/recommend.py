import json
import fasttext
import os
from bpe_tokenizer_for_model import bpe_tokenize, train_tokenizer, load_tokenizer

# 모델 경로 및 데이터 경로 설정 (상대 경로로 수정)
model_path = "model_result/tokenized_product_category_from_fullset_model_epoch15.bin"
data_path = "train_test_data/full_output/before_processed/for_bpe"
tokenizer_path = "tokenizer.json"

# FastText 모델 로드
if os.path.exists(model_path):
    model = fasttext.load_model(model_path)
    print("Prediction Model successfully loaded.")
else:
    print("Model not found. Please train the model first.")
    exit()

# 토크나이저 로드 또는 학습
if os.path.exists(tokenizer_path):
    load_tokenizer()
    print("BPE Tokenizer successfully loaded.")
else:
    train_tokenizer(data_path)
    print("BPE Tokenizer successfully trained")

# Vercel 서버리스 함수
def handler(event, context):
    # 쿼리 파라미터 가져오기
    query = event.get("queryStringParameters", {}).get("query", "")

    if query:
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
    else:
        recommendations = []

    # 응답 반환
    return {
        "statusCode": 200,
        "body": json.dumps({"recommendations": recommendations}),
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        }
    }
