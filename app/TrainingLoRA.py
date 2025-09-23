from .config import CHAT_DATASET_PATH
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import datasets
import os
from datetime import datetime, timedelta
from pathlib import Path
import re
import serial

import app.state as state
from .config import TOKENIZER_PATH

def modelUpdate(chatLog_len):
    start = chatLog_len - 30
    end = chatLog_len
    model_path = FiveFinetuning(start, end)
     
    # base + LoRA をロード
    base_model = AutoModelForCausalLM.from_pretrained(
        "cyberagent/open-calm-small",
        torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(base_model, model_path)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    
    state.model = model
    state.tokenizer = tokenizer
    
    print(f"対話モデルを更新しました: {model_path}")
    state.ser.write(b"TuningEnd\n")
    

def FiveFinetuning(startIndex, endIndex):
    base_dir = Path("./tunedModels")
    today = datetime.now()
    today_str = today.strftime("%Y-%m-%d")

    # 保存先を決める
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

    # テンプレート
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

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    train_dataset = MyDataset(datalist, tokenizer)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    output_dir.mkdir(parents=True, exist_ok=True)

    # LoRA 化
    base_model = AutoModelForCausalLM.from_pretrained(
        "cyberagent/open-calm-small",
        torch_dtype=torch.bfloat16
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    model = get_peft_model(base_model, lora_config)

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

    # LoRA のみ保存
    model.save_pretrained(output_dir / "lora")
    tokenizer.save_pretrained(output_dir / "lora")

    print(f"✅ LoRA アダプタを保存しました: {output_dir}/lora")
    return output_dir / "lora"
