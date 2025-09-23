from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pathlib import Path

def generateAnswer(tokenizer, model, query):
    
    # template = {
    #     "w_input": (
    #         "以下はタスクを記述した指示と入力です。入力はタスクで参照されている文章です。指示を適切に満たす応答を書きなさい。\n\n"
    #         "### 指示:{instruction}\n"
    #         "### 入力:\n{input}\n"
    #         "### 応答:\n"
    #     ),
    #     "wo_input": (
    #         "以下はタスクを記述した指示と入力です。入力はタスクで参照されている文章です。指示を適切に満たす応答を書きなさい。\n\n"
    #         "### 指示:{instruction}\n"
    #         "### 応答:\n"
    #     )
    # }
    
    # d = {}
    # d['instruction'] = query
    # d['output'] = ""
    
    ptext = query
    
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
            max_new_tokens=256,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            top_p=0.9,
        )
        
    import re
    
    response = tokenizer.decode(tokens[0][start_pos:], skip_special_tokens=True)
    print(response)
    
    return response

