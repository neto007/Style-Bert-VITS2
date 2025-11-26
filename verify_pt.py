import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import clean_text

text = "Olá, mundo! Isso é um teste."
try:
    norm_text, phones, tones, word2ph = clean_text(text, Languages.PT)
    print(f"Norm: {norm_text}")
    print(f"Phones: {phones}")
    print(f"Tones: {tones}")
    print(f"Word2Ph: {word2ph}")
    print("Verification Successful!")
except Exception as e:
    print(f"Verification Failed: {e}")
    import traceback
    traceback.print_exc()
