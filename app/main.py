from fastapi import FastAPI, Form, Request, Response, Cookie
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uuid

from .modelLoader import loadModel
from .dialogue import generateQuestion, generateResponse
from .chatLog import saveJSON

app = FastAPI()
templates = Jinja2Templates(directory="./app/templates")
app.mount("/static", StaticFiles(directory="./app/static"), name="static")

model, tokenizer = loadModel()

# セッションごとのチャットログ
session_logs = {}

# セッションIDを取得または生成
def get_or_create_session_id(session_id):
    if not session_id:
        return str(uuid.uuid4())
    return session_id

# 初回かどうかもセッションごとに管理
session_first_request = {}

# GET: 質問を生成
@app.get("/", response_class=HTMLResponse)
def form_get(request: Request, response: Response, session_id: str = Cookie(default=None)):
    session_id = get_or_create_session_id(session_id)
    response.set_cookie(key="session_id", value=session_id)

    question = generateQuestion(tokenizer, model)

    # セッションごとのログと初回フラグを初期化
    session_logs.setdefault(session_id, [])
    session_first_request.setdefault(session_id, True)

    return templates.TemplateResponse("form.html", {
        "request": request,
        "question": question,
        "chatLogs": session_logs[session_id]
    })

# POST: 回答を受け取って応答を生成
@app.post("/", response_class=HTMLResponse)
def form_post(
    request: Request,
    response: Response,
    answer: str = Form(...),
    question: str = Form(...),
    session_id: str = Cookie(default=None)
):
    session_id = get_or_create_session_id(session_id)
    response.set_cookie(key="session_id", value=session_id)

    chat_log = session_logs.setdefault(session_id, [])
    is_first = session_first_request.get(session_id, True)

    response_text = generateResponse(tokenizer, model, question, answer)

    if is_first:
        chat_log.append({"question": question, "answer": answer, "response": response_text})
        session_first_request[session_id] = False
    else:
        chat_log.append({"answer": answer, "response": response_text})

    saveJSON(question, answer)

    return templates.TemplateResponse("form.html", {
        "request": request,
        "question": response_text,
        "chatLogs": chat_log
    })
