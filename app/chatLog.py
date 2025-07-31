#jsonファイルに対話を保存する
from .config import CHAT_LOG_PATH
import json
from pathlib import Path
from datetime import datetime

count = []

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
    print(f"現在の対話数: {len(chatLog)}")

    return len(chatLog)  # ← ここで件数を返す！
    
    
    # # -----ここから10件ごとにファインチューニング-----
    # count.append(newEntry)
    # print(f"現在の対話数: {len(count)}")
    
    # # 10の倍数になったらファインチューニング
    # if len(count) % 10 == 0:
    #     endIndex = len(count)
    #     startIndex = endIndex - 10
        
    #     from .fineTune_5datasets import FiveFinetuning
    #     FiveFinetuning(startIndex, endIndex)