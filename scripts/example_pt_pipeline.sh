#!/usr/bin/env bash
set -euo pipefail

# Example pipeline for PT-BR training with a minimal dataset

MODEL_NAME="pt_demo"

# 1) Slice raw wavs into segments
python slice.py --model_name "$MODEL_NAME"

# 2) Transcribe to build esd.list
python transcribe.py --model_name "$MODEL_NAME" --language PT

# 3) Preprocess text with PT-Extra
python preprocess_text.py --model_name "$MODEL_NAME" --use_pt_extra

# 4) Generate BERT features
python bert_gen.py --model_name "$MODEL_NAME"

# 5) Generate style vectors
python style_gen.py --model_name "$MODEL_NAME" --num_processes 2

# 6) Train for a few steps (adjust config in Data/<MODEL_NAME>/config.json)
python train_ms.py -c "Data/$MODEL_NAME/config.json" --no_progress_bar

