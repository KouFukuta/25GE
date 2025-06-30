#ファインチューニング
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

from .config import TOKENIZER_PATH

def startFinetuning():
    # 昨日の日付の文字列を取得
    yesterday_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    save_path = Path(f"./tunedModels/{yesterday_str}")

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
            local_files_only=True,
            torch_dtype=torch.bfloat16
        )
    else:
        # 昨日のモデルが見つからなかった場合、firstModel を使う
        fallback_checkpoint = Path("./tunedModels/firstModel/checkpoint-500")
        if fallback_checkpoint.exists():
            print(f"No model for yesterday. Loading fallback model: {fallback_checkpoint}")
            model = AutoModelForCausalLM.from_pretrained(
                fallback_checkpoint,
                local_files_only=True,
                torch_dtype=torch.bfloat16
            )
    
    # #モデルの読み込み
    # model_name = "cyberagent/open-calm-small"
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     torch_dtype=torch.bfloat16,
    # )
    
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH
    )
    
    # 昨日のデータセットの読み込み
    yesterday_data_path = Path(f"./chatLog/{yesterday_str}")
    if not yesterday_data_path.exists():
        # 昨日のデータセットが存在しない場合はエラーメッセージを表示
        print(f"Yesterday's data not found: {yesterday_data_path}")
        return
    query = datasets.load_dataset("json", data_files=yesterday_data_path)

    # テンプレートの設定
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
    
    # データリストの作成
    datalist = []
    for i in range(len(query['train'])):
        d = query ['train'][i]
        if (d['input'] == ''):
            ptext = template['wo_input'].format_map(d)
        else:
            ptext = template['w_input'].format_map(d)
        if (len(ptext) < 1500):
            datalist.append(ptext)

            
    from torch.utils.data import Dataset

    class MyDataset(Dataset):
        def __init__(self, datalist, tokenizer):
            self.tokenizer = tokenizer
            self.features = []
            for ptext in datalist:
                input_ids = self.tokenizer.encode(ptext)
                input_ids = input_ids + [ self.tokenizer.eos_token_id ]
                input_ids = torch.LongTensor(input_ids)
                self.features.append({'input_ids': input_ids})
        def __len__(self):
            return len(self.features)
        def __getitem__(self, idx):
            return self.features[idx]

    train_dataset = MyDataset(datalist, tokenizer)
    
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    # 日付付きの保存パス
    today_str = datetime.now().strftime("%Y-%m-%d")
    save_path = f"./tunedModels/{today_str}"

    training_args = TrainingArguments(
        output_dir=save_path,
        num_train_epochs=5,    
        save_steps=2000,
        per_device_train_batch_size=1
    )

    trainer = Trainer(
        model=model,
        data_collator=collator,
        args=training_args,
        train_dataset=train_dataset
    )

    trainer.train()
    print("Successfully trained the new model.")