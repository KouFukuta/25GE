from .config import CHAT_DATASET_PATH
from transformers import TrainingArguments, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import datasets
import os
from datetime import datetime, timedelta
from pathlib import Path
import re

import app.state as state
from .config import TOKENIZER_PATH

def modelUpdate(chatLog_len):
    start = chatLog_len - 10
    end = chatLog_len
    model_path = FiveFinetuning(start, end)
     
    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16
        )
     
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH
     )
    
    state.model = model
    state.tokenizer = tokenizer
    
    print(f"対話モデルを更新しました: {model_path}")
    

def FiveFinetuning(startIndex, endIndex):
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


    # tokenizer読み込み
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    # 次のmodel_num（今日の日付フォルダ＋番号）を決める
    def find_next_model_num(base_dir: Path, date_str: str) -> int:
        numbers = []
        pattern = re.compile(rf"{re.escape(date_str)}-(\d+)")
        for d in base_dir.iterdir():
            if d.is_dir():
                m = pattern.fullmatch(d.name)
                if m:
                    numbers.append(int(m.group(1)))
        return max(numbers) + 1 if numbers else 1

    next_model_num = find_next_model_num(base_dir, today_str)
    output_dir = base_dir / f"{today_str}-{next_model_num}"

    # データセット読み込み
    dataset_path = Path(f"./chatLog/{today_str}.json")
    if not dataset_path.exists():
        print(f"📁 今日のデータが見つかりません: {dataset_path}")
        return

    dataset = datasets.load_dataset("json", data_files=str(dataset_path))["train"]
    dataset = dataset.select(range(startIndex, endIndex))

    # テンプレート生成
    template = {
        "w_input": (
            "以下はタスクを記述した指示と入力です。入力はタスクで参照されている文章です。指示を適切に満たす応答を書きなさい。\n\n"
            "### 指示:\n{instruction}\n\n"
            "### 入力:\n{input}\n\n"
            "### 応答:\n{output}"
        ),
        "wo_input": (
            "以下はタスクを記述した指示と入力です。入力はタスクで参照されている文章です。指示を適切に満たす応答を書きなさい。\n\n"
            "### 指示:\n{instruction}\n\n"
            "### 応答:\n{output}"
        )
    }

    datalist = []
    for i in range(len(dataset)):
        d = dataset[i]
        if (d.get('input', '') == ''):
            ptext = template['wo_input'].format_map(d)
        else:
            ptext = template['w_input'].format_map(d)
        if (len(ptext) < 1500):
            datalist.append(ptext)

    class MyDataset(torch.utils.data.Dataset):
        def __init__(self, texts, tokenizer):
            self.features = []
            for text in texts:
                ids = tokenizer.encode(text) + [tokenizer.eos_token_id]
                self.features.append({"input_ids": torch.LongTensor(ids)})

        def __len__(self): return len(self.features)
        def __getitem__(self, idx): return self.features[idx]

    train_dataset = MyDataset(datalist, tokenizer)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=5,
        save_steps=2000,
        per_device_train_batch_size=1
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset
    )

    trainer.train()
    print(f"✅ モデルを保存しました: {output_dir}")
    
    # ここでoutput_dir内の checkpoint-* フォルダを探して最大番号を返す関数を作成
    def get_latest_checkpoint(folder: Path):
        checkpoints = []
        pattern = re.compile(r"checkpoint-(\d+)")
        for d in folder.iterdir():
            if d.is_dir():
                m = pattern.fullmatch(d.name)
                if m:
                    checkpoints.append((int(m.group(1)), d))
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda x: x[0])[1]

    latest_checkpoint = get_latest_checkpoint(output_dir)
    
    print(f"最新のチェックポイント: {latest_checkpoint}")
    return latest_checkpoint