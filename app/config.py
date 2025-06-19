from pathlib import Path
import torch

# モデル関連
# MODEL_PATH = "rinna/japanese-gpt2-medium"
MODEL_PATH = str(Path("./tunedModels/output/checkpoint-500/").resolve())
TOKENIZER_PATH = "cyberagent/open-calm-small"
TOKENIZER_USE_FAST = False
TRUST_REMOTE_CODE = True
LOCAL_FILES_ONLY = True

CHAT_DATASET_PATH = Path("../chatLog/chatLog.json").resolve()

# チャットログの保存場所
CHAT_LOG_PATH = Path("chatLog/chatLog.json")
