from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pathlib import Path

# ユーザーに対する質問を作成


def generateQuestion(tokenizer, model):
    template = {
        "w_input": (
            "以下はタスクを記述した指示と入力です。入力はタスクで参照されている文章です。指示を適切に満たす応答を書きなさい。\n\n"
            "### 指示:\n{instruction}\n\n"
            "### 入力:\n{input}\n\n"
            "### 応答:\n"
        ),
        "wo_input": (
            "以下はタスクを記述した指示と入力です。入力はタスクで参照されている文章です。指示を適切に満たす応答を書きなさい。\n\n"
            "### 指示:\n{instruction}\n\n"
            "### 応答:\n"
        )
    }

    d = {}
    d['instruction'] = "質問を作成して"
    d['output'] = ""

    ptext = template['wo_input'].format_map(d)

    inputs = tokenizer.encode_plus(
        ptext,
        return_tensors="pt",
        padding=True,
    )
    start_pos = inputs['input_ids'].shape[1]
    with torch.no_grad():
        print(f"Starting generation, input shape: {inputs['input_ids'].shape}")
        tokens = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=64,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            top_p=0.9,
        )
        print("Generated token IDs:", tokens)  # ★ここ追加！

    query = tokenizer.decode(tokens[0][start_pos:], skip_special_tokens=True)
    print(query)

    return query


def generateResponse(tokenizer, model, query, answer):
    # ユーザーに質問を投げる
    template = {
        "w_input": (
            "以下はユーザーとの会話の続きです。共感をしながら内容を深堀りしてください。\n\n"
            "{query}\n"
            
            "### 指示:\n{instruction}\n\n"
            "### 入力:\n{input}\n\n"
            "### 応答:\n"
        ),
        "wo_input": (
            "以下はユーザーとの会話の続きです。共感をしながら内容を深堀りしてください。\n\n"
            "{query}\n"
            "### 指示:\n{instruction}\n\n"
            "### 応答:\n"
        )
    }
    
    d = {
        "query": query
    }
    d['instruction'] = answer
    d['output'] = ""

    ptext = template['wo_input'].format_map(d)

    inputs = tokenizer.encode_plus(
        ptext,
        return_tensors="pt",
        padding=True,
    )
    start_pos = inputs['input_ids'].shape[1]
    with torch.no_grad():
        tokens = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=64,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            top_p=0.9,
        )

    response = tokenizer.decode(tokens[0][start_pos:], skip_special_tokens=True)

    import re
    response = re.sub(r"###.*", "", response).strip()

    print(response)

    return response