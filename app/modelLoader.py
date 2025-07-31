# モデルのインプット
from .config import MODEL_PATH, TOKENIZER_PATH
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pathlib import Path
from datetime import datetime, timedelta
import re

import app.state as state
from .fineTune_5datasets import FiveFinetuning

def loadModelForQuestion():
    Qmodel = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )
    Qtokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH
    )
    print("Successfully loaded question Model and tokenizer.")

    return Qmodel, Qtokenizer


def loadModel():
    base_dir = Path("./tunedModels")
    today = datetime.now()
    today_str = today.strftime("%Y-%m-%d")
    yesterday_str = (today - timedelta(days=1)).strftime("%Y-%m-%d")

    def find_latest_model_num_folder(base_dir: Path, date_str: str):
        if not base_dir.exists():
            return None
        model_folders = []
        pattern = re.compile(rf"{re.escape(date_str)}-(\d+)")
        for d in base_dir.iterdir():
            if d.is_dir():
                m = pattern.fullmatch(d.name)
                if m:
                    model_folders.append((int(m.group(1)), d))
        if not model_folders:
            return None
        return max(model_folders, key=lambda x: x[0])[1]

    # まず今日のモデルを探す
    model_num_folder = find_latest_model_num_folder(base_dir, today_str)

    # なければ昨日を探す
    if model_num_folder is None:
        print("今日のモデルが見つかりません。昨日のモデルを探します。")
        model_num_folder = find_latest_model_num_folder(base_dir, yesterday_str)

    if model_num_folder is None:
        print("モデルフォルダが見つかりません。open-calm-smallを使用します。")
        model = AutoModelForCausalLM.from_pretrained(
            "cyberagent/open-calm-small",
            torch_dtype=torch.bfloat16
        )
    else:
        # チェックポイント探す
        def find_latest_checkpoint_folder(model_folder: Path):
            checkpoints = []
            pattern = re.compile(r"checkpoint-(\d+)")
            for d in model_folder.iterdir():
                if d.is_dir():
                    m = pattern.fullmatch(d.name)
                    if m:
                        checkpoints.append((int(m.group(1)), d))
            if not checkpoints:
                return None
            return max(checkpoints, key=lambda x: x[0])[1]

        checkpoint_folder = find_latest_checkpoint_folder(model_num_folder)

        if checkpoint_folder is None:
            print(f"チェックポイントが見つかりません。フォルダから直接ロード: {model_num_folder}")
            model = AutoModelForCausalLM.from_pretrained(
                str(model_num_folder),
                local_files_only=True,
                torch_dtype=torch.bfloat16
            )
        else:
            print(f"📦 モデル読み込み: {checkpoint_folder}")
            model = AutoModelForCausalLM.from_pretrained(
                str(checkpoint_folder),
                local_files_only=True,
                torch_dtype=torch.bfloat16
            )

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH
    )
    print("Successfully loaded model and tokenizer.")
    print(f"Pad token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")
    print(f"EOS token: {tokenizer.eos_token}, ID: {tokenizer.eos_token_id}")

    return model, tokenizer