# モデルのインプット
from .config import MODEL_PATH, TUNED_MODEL_PATH
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pathlib import Path


def loadModel():

    # モデルとトークナイザーの読み込み
    model = AutoModelForCausalLM.from_pretrained(
        "./output/checkpoint-500",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "cyberagent/open-calm-small",
    )

    print("Successfully loaded model and tokenizer.")

    return tokenizer, model
