from transformers import AutoModelForMaskedLM, AutoTokenizer
import os
from pathlib import Path

# Define path
base_dir = Path("/home/machine/repository/google_audio/Style-Bert-VITS2/style_bert_vits2/bert/bert-base-portuguese-cased")
base_dir.mkdir(parents=True, exist_ok=True)

model_name = "neuralmind/bert-base-portuguese-cased"

print(f"Downloading {model_name} to {base_dir}...")

# Download and save tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(base_dir)

# Download and save model
model = AutoModelForMaskedLM.from_pretrained(model_name)
model.save_pretrained(base_dir)

print("Download complete.")
