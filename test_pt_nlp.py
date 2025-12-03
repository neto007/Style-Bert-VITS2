import torch
from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import clean_text, extract_bert_feature

def test_pt_nlp():
    text = "Olá, mundo! O número é 42."
    print(f"Texto original: {text}")
    
    # Teste de Normalização e G2P
    try:
        norm_text, phones, tones, word2ph = clean_text(text, Languages.PT)
        print(f"Texto normalizado: {norm_text}")
        print(f"Fonemas: {phones}")
        print(f"Word2Ph: {word2ph}")
    except Exception as e:
        print(f"Erro no G2P: {e}")
        return

    # Teste de BERT
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Usando dispositivo: {device}")
        bert_feature = extract_bert_feature(text, word2ph, Languages.PT, device)
        print(f"BERT feature shape: {bert_feature.shape}")
    except Exception as e:
        print(f"Erro no BERT: {e}")

if __name__ == "__main__":
    test_pt_nlp()
