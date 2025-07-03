from fastapi import FastAPI, Form, Request, Response, Cookie
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uuid
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
from pytz import timezone
from contextlib import asynccontextmanager

from .modelLoader import loadModel
from .dialogue import generateQuestion, generateResponse
from .chatLog import saveJSON
from .fineTune import startFinetuning

# スケジューラの初期化
scheduler = BackgroundScheduler()
scheduler.configure(timezone=timezone("Asia/Tokyo"))

# ファインチューニングを毎日0時に実行
def scheduled_finetune_job():
    print("starting scheduled fine-tuning job...")
    save_path = startFinetuning()
    print(f"saved model: {save_path}")

    new_model, new_tokenizer = loadModel()
    update_model_tokenizer(new_model, new_tokenizer)

    print("updated model and tokenizer！")

def update_model_tokenizer(new_model, new_tokenizer):
    global model, tokenizer
    model = new_model
    tokenizer = new_tokenizer

scheduler.add_job(scheduled_finetune_job, 'cron', hour=20, minute=28)


# FastAPIを lifespan付きで最初から定義
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("0:00 Starting scheduled job for fine-tuning...")
    scheduler.start()
    yield
    scheduler.shutdown()


app = FastAPI(lifespan=lifespan)
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
def form_get(request: Request, session_id: str = None):
    # セッションIDがURLパラメータにない場合は、新規生成してリダイレクト
    if not session_id:
        new_id = str(uuid.uuid4())
        return RedirectResponse(url=f"/?session_id={new_id}")
    
    # ログ管理
    session_logs.setdefault(session_id, [])
    session_first_request.setdefault(session_id, True)

    question = generateQuestion(tokenizer, model)

    return templates.TemplateResponse("form.html", {
        "request": request,
        "question": question,
        "chatLogs": session_logs[session_id],
        "session_id": session_id,  # テンプレートにも渡す
    })



# POST: 回答を受け取って応答を生成
@app.post("/", response_class=HTMLResponse)
def form_post(
    request: Request,
    answer: str = Form(...),
    question: str = Form(...),
    session_id: str = Form(...),  # URLじゃなくフォームのhiddenから
):
    chat_log = session_logs.setdefault(session_id, [])
    recent_logs = chat_log[-5:]
    history_text = ""
    for log in recent_logs:
        history_text += f"Question: {log['question']}\n"
        history_text += f"User: {log['answer']}\n"
        history_text += f"AI: {log['response']}\n"
    
    # 今回の質問を履歴に加える
    full_context = f"{history_text}Question: {question}\n"

    response_text = generateResponse(tokenizer, model, full_context, answer)

    # 履歴に今回のやりとりを追加
    chat_log.append({
        "question": question,
        "answer": answer,
        "response": response_text,
    })

    saveJSON(question, answer)

    return templates.TemplateResponse("form.html", {
        "request": request,
        "question": response_text,
        "chatLogs": chat_log,
        "session_id": session_id,
    })
