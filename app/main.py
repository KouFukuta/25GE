import app.config as config
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from .modelLoader import loadModel
from .dialogue import generateQuestion, generateResponse
from .chatLog import saveJSON

app = FastAPI()
templates = Jinja2Templates(directory="./app/templates")
app.mount("/static", StaticFiles(directory="./app/static"), name="static")

# チャットのログをWeb側に保存するためのリスト
chatLogs = []

# モデルとトークナイザーの読み込み
model, tokenizer = loadModel()


# 質問を生成して送信
@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    question = generateQuestion(tokenizer, model)
    return templates.TemplateResponse("form.html", {
        "request": request,
        "question": question,
        "chatLogs": chatLogs
    })

is_first_request = True

# インプットを受け取って対話を生成
@app.post("/", response_class=HTMLResponse)
def form_post(request: Request, answer: str = Form(...), question: str = Form(...)):
    global is_first_request
    response = generateResponse(tokenizer, model, question, answer)
    
    if is_first_request:
        chatLogs.append({"question": question, "answer": answer, "response": response})
        is_first_request = False
    else:
        chatLogs.append({"answer": answer, "response": response})
    
    saveJSON(question, answer)

    
    return templates.TemplateResponse("form.html", {
        "request": request,
        "question": response,
        "chatLogs": chatLogs 
    })
