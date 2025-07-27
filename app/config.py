from pathlib import Path
import torch

#　元モデル
# MODEL_PATH = str(Path("./tunedModels/output/checkpoint-500/").resolve())

# 深掘りさせるモデル
# MODEL_PATH = str(Path("./tunedModels/prototype_fukabori/checkpoint-150/").resolve())

# 深掘りを対話形式で学習させたモデル
# MODEL_PATH = str(Path("./tunedModels/prototype_fukaboriII/checkpoint-500/").resolve())

# IとIIを統合したモデル
# MODEL_PATH = str(Path("./tunedModels/prototype_fukaboriIII/checkpoint-500/").resolve())

MODEL_PATH = str(Path("./tunedModels/kariModels/kari1-3/checkpoint-1080/").resolve())

TOKENIZER_PATH = "cyberagent/open-calm-small"
TOKENIZER_USE_FAST = False
TRUST_REMOTE_CODE = True
LOCAL_FILES_ONLY = True

CHAT_DATASET_PATH = str("./chatLog/chatLog.json")

# チャットログの保存場所
CHAT_LOG_PATH = Path("chatLog/chatLog.json")
