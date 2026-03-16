# AI-Mathematical-Olympiad

## OpenMathReasoning QLoRA

Train a 7B LoRA adapter (internet ON):

```bash
pip install -q --no-cache-dir "transformers>=4.44.0" datasets peft trl bitsandbytes accelerate
python scripts/train_qlora_openreasoning.py \
  --base-model Qwen/Qwen2.5-Math-7B-Instruct \
  --output-dir /kaggle/working/orm-7b-lora \
  --max-samples 60000
```

For inference in `notebooks/aimo_3_in_progress.ipynb` (internet OFF), attach the adapter as a Kaggle dataset and set:

- `USE_LORA_ADAPTER = True`
- `LORA_ADAPTER_PATH = "/kaggle/input/<your-adapter-dataset-folder>"`
