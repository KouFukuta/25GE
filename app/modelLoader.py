# モデルのインプット
from .config import TOKENIZER_PATH
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pathlib import Path
from datetime import datetime
import re

def loadModel():
    today_str = datetime.now().strftime("%Y-%m-%d")
    save_path = Path(f"./tunedModels/{today_str}")

    last_checkpoint = None
    if save_path.exists():
        checkpoints = []
        for d in save_path.glob("checkpoint-*"):
            if d.is_dir():
                m = re.search(r'checkpoint-(\d+)', d.name)
                if m:
                    checkpoints.append((int(m.group(1)), d))

        if checkpoints:
            last_checkpoint = max(checkpoints, key=lambda x: x[0])[1]

    if last_checkpoint:
        print(f"Loading fine-tuned model from: {last_checkpoint}")
        model = AutoModelForCausalLM.from_pretrained(
            last_checkpoint,
            local_files_only=True
        )
    else:
        # 今日のファインチューニングモデルが見つからなかった場合、firstModel を使う
        fallback_checkpoint = Path("./tunedModels/firstModel/checkpoint-500")
        if fallback_checkpoint.exists():
            print(f"No model for today. Loading fallback model: {fallback_checkpoint}")
            model = AutoModelForCausalLM.from_pretrained(
                fallback_checkpoint,
                local_files_only=True
            )
        else:
            raise FileNotFoundError(
                f"Neither today's model nor fallback checkpoint found. "
                f"Looked for: {save_path} and {fallback_checkpoint}"
            )

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH
    )
    print("Successfully loaded model and tokenizer.")
    print(f"Pad token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")
    print(f"EOS token: {tokenizer.eos_token}, ID: {tokenizer.eos_token_id}")

    return model, tokenizer
