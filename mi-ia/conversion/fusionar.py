import os
import gc
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL   = "Qwen/Qwen2.5-1.5B"
ADAPTER_PATH = "./pokemon-qwen-lora"
OUTPUT_PATH  = "./pokemon-qwen-fusionado"

def main():
    if not os.path.isdir(ADAPTER_PATH):
        raise FileNotFoundError(f"No existe la carpeta del adaptador: {ADAPTER_PATH}")

    print("Cargando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    print("Cargando modelo base...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float32,
        trust_remote_code=True,
    )

    print("Cargando adaptador LoRA...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    print("Fusionando...")
    merged_model = model.merge_and_unload()

    del model
    del base_model
    gc.collect()

    print(f"Guardando en: {OUTPUT_PATH}")
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    merged_model.save_pretrained(OUTPUT_PATH, safe_serialization=True, max_shard_size="500MB")
    tokenizer.save_pretrained(OUTPUT_PATH)

    print("=" * 50)
    print("FUSIÓN COMPLETADA")
    print(f"Modelo guardado en: {OUTPUT_PATH}")
    print("Siguiente paso: ./convertir_gguf.sh")
    print("=" * 50)

if __name__ == "__main__":
    main()
