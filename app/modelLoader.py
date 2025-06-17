# モデルのインプット
from .config import MODEL_PATH, TOKENIZER_PATH
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pathlib import Path


def loadModel():

    # モデルとトークナイザーïïの読み込み
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only = True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH
    )

    print("Successfully loaded model and tokenizer.")

    return model, tokenizer