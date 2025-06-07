#jsonファイルに対話を保存する
from .config import CHAT_LOG_PATH
import json
from pathlib import Path

def saveJSON(input, output):
    file = CHAT_LOG_PATH
    
    if file.exists():
        with open(file, "r", encoding="utf-8") as f:
            chatLog = json.load(f)
            
    else:
        chatLog = []
        
    newEntry = {
        "instruction": input,
        "output": output
    }
    
    chatLog.append(newEntry)
    
    with open(file, "w", encoding="utf-8") as f:
        json.dump(chatLog, f, ensure_ascii=False, indent=2)
        
    print("対話をセーブしました")
    