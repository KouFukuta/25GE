#ファインチューニング
from .config import LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_BIAS, DEVICE, TRUNCATION, MAX_LENGTH, PADDING, OUTPUT_DIR, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE, GRAD_ACCUM_STEPS, SAVE_TOTAL_LIMIT, LOGGING_STEPS, CHAT_DATASET_PATH
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

def startFinetuning(tokenizer, model):
    
    #LoRAの設定
    peftConfig = LoraConfig(
        r = LORA_R,
        lora_alpha = LORA_ALPHA,
        task_type = TaskType.CAUSAL_LM,
        lora_dropout = LORA_DROPOUT,
        bias = LORA_BIAS
    )
    
    model = get_peft_model(model, peftConfig)

    datasets = load_dataset("json", data_files = CHAT_DATASET_PATH)["train"]

    def tokenize(examples):
        return tokenizer(
            examples["instruction"],
            text_target = examples["output"],
            truncation = TRUNCATION,
            max_length = MAX_LENGTH,
            padding = PADDING,
        )
        
    trainArgs = TrainingArguments(
        output_dir = OUTPUT_DIR,
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM_STEPS,
        num_train_epochs = NUM_EPOCHS,
        fp16 = True if DEVICE.type == "cuda" else False,
        save_total_limit = SAVE_TOTAL_LIMIT,
        logging_steps = LOGGING_STEPS,
        learning_rate = LEARNING_RATE,
    )

    trainer = Trainer(
        model = model,
        args = trainArgs,
        train_dataset=datasets.map(tokenize, batched=True, remove_columns=["instruction", "output"]),
        tokenizer = tokenizer,
        data_collator = None
    )

    trainer.train()

    #モデルの保存
    model.save_pretrained("./tunedModel")
    tokenizer.save_pretrained("./tunedModel")
