# bpe_tokenizer.py

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import normalizers
from tokenizers.normalizers import NFKC
import os

# BPE 토크나이저 생성
tokenizer = Tokenizer(BPE())
tokenizer.normalizer = normalizers.Sequence([NFKC()])
tokenizer.pre_tokenizer = Whitespace()

# BPE 트레이너 설정
trainer = BpeTrainer(special_tokens=["<unk>"], vocab_size=30000)

def train_tokenizer(data_path):
    # 학습 데이터로 BPE 학습
    files = [os.path.join(data_path, file_name) for file_name in os.listdir(data_path)]
    tokenizer.train(files, trainer)
    tokenizer.save("tokenizer.json")
    
def bpe_tokenize(text):
    # BPE 토크나이징 수행
    output = tokenizer.encode(text)
    # print(f'Encoded Output: {output}')
    # print(f'Tokens: {output.tokens}')
    # 공백으로 구분된 토큰을 반환
    return ' '.join(output.tokens)

def load_tokenizer():
    # 저장된 모델 로드
    global tokenizer
    tokenizer = Tokenizer.from_file("tokenizer.json")