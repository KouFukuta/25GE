#ファインチューニング
from .config import CHAT_DATASET_PATH
from transformers import TrainingArguments, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import datasets
import os
from datetime import datetime
from pathlib import Path
import re

from .config import TOKENIZER_PATH

def startFinetuning():
    #モデルの読み込み
    model_name = "cyberagent/open-calm-small"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    query = datasets.load_dataset("json", data_files=CHAT_DATASET_PATH)

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