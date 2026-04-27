import os
import json
import random

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


MODEL_NAME   = "Qwen/Qwen2.5-1.5B"
DATA_FILE    = "datos.jsonl"
OUTPUT_DIR   = "pokemon-qwen-lora"

SEED                        = 42
VAL_RATIO                   = 0.05
MAX_LENGTH                  = 768
PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE  = 1
GRADIENT_ACCUMULATION_STEPS = 8
NUM_TRAIN_EPOCHS            = 5
LEARNING_RATE               = 2e-4
WEIGHT_DECAY                = 0.01
WARMUP_RATIO                = 0.03
LOGGING_STEPS               = 10
SAVE_TOTAL_LIMIT            = 2

PROMPT_PREFIX = "### Pregunta:\n"
PROMPT_MIDDLE = "\n\n### Respuesta:\n"

CHECKPOINT = None


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

def build_text(question, answer):
    return f"{PROMPT_PREFIX}{question.strip()}{PROMPT_MIDDLE}{answer.strip()}"

def load_jsonl(filepath):
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[AVISO] Línea {i} inválida: {e}")
                continue
            q = str(item.get("question", "")).strip()
            a = str(item.get("answer", "")).strip()
            if not q or not a:
                print(f"[AVISO] Línea {i} incompleta, se omite")
                continue
            rows.append({"text": build_text(q, a)})
    return rows

def split_dataset(rows, val_ratio=0.05, seed=42):
    rng = random.Random(seed)
    rows = rows[:]
    rng.shuffle(rows)
    val_size = max(1, int(len(rows) * val_ratio))
    return rows[val_size:], rows[:val_size]


# MAIN
def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Cargando datos desde: {DATA_FILE}")
    rows = load_jsonl(DATA_FILE)
    if not rows:
        raise ValueError("No se cargó ningún ejemplo válido.")

    train_rows, eval_rows = split_dataset(rows, VAL_RATIO, SEED)
    train_dataset = Dataset.from_list(train_rows)
    eval_dataset  = Dataset.from_list(eval_rows)

    print(f"Ejemplos de entrenamiento : {len(train_dataset)}")
    print(f"Ejemplos de validación    : {len(eval_dataset)}")
    print(f"Epochs                    : {NUM_TRAIN_EPOCHS}")

    print("Cargando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Cargando modelo base...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    print("Preparando LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        fp16=False,
        bf16=False,
        report_to="none",
        max_length=MAX_LENGTH,
        completion_only_loss=False,
        dataset_text_field="text",
        packing=False,
        seed=SEED,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print("=" * 50)
    print("INICIANDO ENTRENAMIENTO")
    print(f"Epochs     : {NUM_TRAIN_EPOCHS}")
    print(f"Estimación : 4-5 horas en CPU")
    print("=" * 50)

    trainer.train(resume_from_checkpoint=CHECKPOINT)

    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("=" * 50)
    print("ENTRENAMIENTO COMPLETADO")
    print(f"Adaptador guardado en: {OUTPUT_DIR}")
    print("Siguiente paso: python3 fusionar.py")
    print("=" * 50)

if __name__ == "__main__":
    main()
