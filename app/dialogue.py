from .config import DEVICE
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pathlib import Path

#ユーザーに対する質問を作成

def generateQuestion(tokenizer, model):

    prompt = "私に対して短い質問をしてください。"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=50,
        add_special_tokens=True,
    ).to(DEVICE)

    model.eval()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.7,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    query = full_text[len(prompt):]
    
    print("Question: " + query)
    
    return query

def generateResponse(tokenizer, model, answer):
    #ユーザーに質問を投げる
    prompt = answer
    print("Input: " + answer)

    inputs = tokenizer(
        prompt,
        return_tensors = "pt",
        truncation = True,
        max_length = 512,
        add_special_tokens = True,
    ).to(DEVICE)

    model.eval()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = 50,
            pad_token_id = tokenizer.pad_token_id,
            do_sample = True,
            temperature = 0.7,
        )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    generated = full_text[len(prompt):]

    # 結果の出力
    print("Output: " + generated.strip())
    
    return prompt, generated

