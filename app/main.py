import app.config as config
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from .modelLoader import loadModel
from .fineTune import startFinetuning
from .dialogue import generateQuestion, generateResponse
from .chatLog import saveJSON

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# モデルとトークナイザーの読み込み
tokenizer, model = loadModel()

# 質問を生成して送信
@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    question = generateQuestion(tokenizer, model)
    return templates.TemplateResponse("form.html", {"request": request, "question": question})

# インプットを受け取って対話を生成
@app.post("/", response_class=HTMLResponse)
def form_post(request: Request, answer: str = Form(...), question: str = Form(...)):
    
    response = generateResponse(tokenizer, model, answer)
    saveJSON(question, answer)
    return templates.TemplateResponse("form.html", {
        "request": request,
        "question": question,
        "answer": answer,
        "response": response,
    })
