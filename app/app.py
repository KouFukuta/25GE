#!/usr/bin/env python3.11
from fastapi import FastAPI, Request
from pydantic import BaseModel

from .modelLoader import loadModel
from .dialogue import generateDialogue
from .fineTune import startFinetuning


app = FastAPI()

# モデルとトークナイザーを初期化
tokenizer, model = loadModel()

class DialogueRequest(BaseModel):
    query: str

@app.post("/generate")
def generate(dialogue: DialogueRequest):
    response = generateDialogue(tokenizer, model, dialogue.query)
    return {"response": response}

# @app.post("/finetune")
# def finetune():
#     startFinetuning(tokenizer, model)
#     return {"message": "Fine-tuning started."}
