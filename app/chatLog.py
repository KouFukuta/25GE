#jsonファイルに対話を保存する
from .config import CHAT_LOG_PATH
import json
from pathlib import Path
from datetime import datetime

def saveJSON(input, output):
    today_str = datetime.now().strftime("%Y-%m-%d")
    save_path = Path(f"./chatLog/{today_str}.json")
    file = save_path
    
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
    