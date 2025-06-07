#モデルのインプット
from .config import MODEL_PATH, DEVICE, TOKENIZER_USE_FAST, LOCAL_FILES_ONLY, TRUST_REMOTE_CODE, DTYPE
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pathlib import Path

def loadModel():
    # モデルとトークナイザーの読み込み
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        padding_side = 'left',
        use_fast = TOKENIZER_USE_FAST,
        local_files_only = LOCAL_FILES_ONLY,
        trust_remote_code = TRUST_REMOTE_CODE
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype = DTYPE,
        local_files_only = LOCAL_FILES_ONLY,
        trust_remote_code = TRUST_REMOTE_CODE
    ).to(DEVICE)

    print("Successfully loaded model and tokenizer.")
    
    return tokenizer, model
    
    