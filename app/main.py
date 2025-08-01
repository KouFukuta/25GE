from fastapi import FastAPI, Form, Request, Response, Cookie
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uuid
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
from pytz import timezone
from contextlib import asynccontextmanager
import threading
import serial

import app.state as state
from .modelLoader import loadModel, loadModelForQuestion
from .dialogue import generateQuestion, generateResponse
from .chatLog import saveJSON
from .fineTune import startFinetuning
from .generate import generateAnswer
from .fineTune_5datasets import modelUpdate
from .setUSB import findNuigurumi

# # -----ファインチューニングのスケジューラー-----

# # スケジューラの初期化
# scheduler = BackgroundScheduler()
# scheduler.configure(timezone=timezone("Asia/Tokyo"))

# # ファインチューニングを毎日0時に実行
# def scheduled_finetune_job():
#     print("starting scheduled fine-tuning job...")
#     save_path = startFinetuning()
#     print(f"saved model: {save_path}")

#     new_model, new_tokenizer = loadModel()
#     update_model_tokenizer(new_model, new_tokenizer)

#     print("updated model and tokenizer！")

# def update_model_tokenizer(new_model, new_tokenizer):
#     global model, tokenizer
#     model = new_model
#     tokenizer = new_tokenizer

# # ここでファインチューニングの時間設定
# scheduler.add_job(scheduled_finetune_job, 'cron', hour=0, minute=0)


# # FastAPIを lifespan付きで最初から定義
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     print("0:00 Starting scheduled job for fine-tuning...")
#     scheduler.start()
#     yield
#     scheduler.shutdown()

# -----fastAPIの設定-----

# app = FastAPI(lifespan=lifespan)
app = FastAPI()
templates = Jinja2Templates(directory="./app/templates")
app.mount("/static", StaticFiles(directory="./app/static"), name="static")

# 生成で使う、成長させるモデル
state.model, state.tokenizer = loadModel()
# 質問をさせて知識を集めるモデル
Qmodel, Qtokenizer = loadModelForQuestion()

# ファインチューニングを前に行った件数
last_finetune_count = 0

# セッションごとのチャットログ
session_logs = {}

# セッションIDを取得または生成
def get_or_create_session_id(session_id):
    if not session_id:
        return str(uuid.uuid4())
    return session_id

# 初回かどうかもセッションごとに管理
session_first_request = {}

# シリアル通信の設定
state.ser = findNuigurumi()

if state.ser:
    try:
        state.ser.write(b"connected\n")
    except Exception as e:
        print(f"⚠️ シリアル送信エラー: {e}")
else:
    print("ぬいぐるみ無しでも動作続行します〜")



# -----form.htmlで使うコード-----

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

    question = generateQuestion(Qtokenizer, Qmodel)

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
    global last_finetune_count
    
    state.ser.write(b'response\n')
    
    chat_log = session_logs.setdefault(session_id, [])
    recent_logs = chat_log[-5:]
    history_text = ""
    for log in recent_logs:
        history_text += f"Question: {log['question']}\n"
        history_text += f"User: {log['answer']}\n"
        history_text += f"AI: {log['response']}\n"
    
    # 今回の質問を履歴に加える
    full_context = f"{history_text}Question: {question}\n"

    response_text = generateResponse(Qtokenizer, Qmodel, full_context, answer)
    
    if not chat_log:
        # 最初の質問と人間の回答
        chatLog_len = saveJSON(question, answer)
    else:
        # 前回の人間の回答に対するAIの質問
        prev = chat_log[-1]
        chatLog_len = saveJSON(prev["answer"], question)  # instruction: 人間の答え, output: 今回のAI質問
        chatLog_len = saveJSON(question, answer)    
        # instruction: AI質問, output: 人間の答え
        
    # ファインチューニングのトリガー
    if chatLog_len - last_finetune_count >= 10:
        state.ser.write(b"Tuning\n")
        threading.Thread(target=modelUpdate, args=(chatLog_len,)).start()
        last_finetune_count = chatLog_len  # 実行後に更新

    # 履歴に今回のやりとりを追加
    chat_log.append({
        "question": question,
        "answer": answer,
        "response": response_text,
    })
    
    state.ser.write(b'finishResponse\n')

    return templates.TemplateResponse("form.html", {
        "request": request,
        "question": response_text,
        "chatLogs": chat_log,
        "session_id": session_id,
    })
    
# 最初の質問の再生成 答えられない時など
@app.post("/reGenerate", response_class=HTMLResponse)
def reGenerateQuestion(request: Request, 
                       session_id: str = Form(...)
                    ):
    # セッションIDがURLパラメータにない場合は、新規生成してリダイレクト
    if not session_id:
        new_id = str(uuid.uuid4())
        return RedirectResponse(url=f"/?session_id={new_id}")
    
    # ログ管理
    session_logs.setdefault(session_id, [])
    session_first_request.setdefault(session_id, True)

    question = generateQuestion(Qtokenizer, Qmodel)

    return templates.TemplateResponse("form.html", {
        "request": request,
        "question": question,
        "chatLogs": session_logs[session_id],
        "session_id": session_id,  # テンプレートにも渡す
    })
    
    
    
# -----普通のAIとして使う場所-----

@app.get("/generate", response_class=HTMLResponse)
def generate_get(request: Request, session_id: str = None):
    # セッションIDがURLパラメータにない場合は、新規生成してリダイレクト
    if not session_id:
        new_id = str(uuid.uuid4())
        return RedirectResponse(url=f"/generate?session_id={new_id}")

    # ログ管理
    session_logs.setdefault(session_id, [])
    session_first_request.setdefault(session_id, True)

    return templates.TemplateResponse("generate.html", {
        "request": request,
        "chatLogs": session_logs[session_id],
        "session_id": session_id,  # テンプレートにも渡す
    })

@app.post("/generate", response_class=HTMLResponse)
def generate_post(
    request: Request,
    question: str = Form(...),
    session_id: str = Form(...),  # URLじゃなくフォームのhiddenから
):
    Gen_chat_log = session_logs.setdefault(session_id, [])
    
    # 履歴を生成
    history_text = ""
    for log in Gen_chat_log:
        history_text += f"Question: {log['question']}\n"
        history_text += f"User: {log['answer']}\n"

    response_text = generateAnswer(state.tokenizer, state.model, question)

    # 履歴に今回のやりとりを追加
    Gen_chat_log.append({
        "question": question,
        "answer": response_text,
    })

    return templates.TemplateResponse("generate.html", {
        "request": request,
        "response": response_text,
        "chatLogs": Gen_chat_log,
        "session_id": session_id,
    })