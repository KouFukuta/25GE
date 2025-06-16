from pathlib import Path
import torch

# モデル関連
# MODEL_PATH = "rinna/japanese-gpt2-medium"
MODEL_PATH = Path(r"./output/checkpoint-500")
TUNED_MODEL_PATH = Path("cyberagent/open-calm-small")
TOKENIZER_USE_FAST = False
TRUST_REMOTE_CODE = True
LOCAL_FILES_ONLY = True

# デバイス設定
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DTYPE = torch.float16
    DEVICE_NAME = torch.cuda.get_device_name(0)
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    DTYPE = torch.float32
    DEVICE_NAME = "Apple Silicon (MPS)"
else:
    DEVICE = torch.device("cpu")
    DTYPE = torch.float32
    DEVICE_NAME = "CPU"

# 学習関連
OUTPUT_DIR = Path("./TunedModel").resolve()
NUM_EPOCHS = 10
LEARNING_RATE = 5e-5
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
SAVE_TOTAL_LIMIT = 2
LOGGING_STEPS = 10

CHAT_DATASET_PATH = Path("../chatLog/chatLog.json").resolve()

# LoRA設定
LORA_R = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_BIAS = "none"

# LoRAトークナイズ設定
TRUNCATION = True
MAX_LENGTH = 512
PADDING = "max_length"


# チャットログの保存場所
CHAT_LOG_PATH = Path("chatLog/chatLog.json")
