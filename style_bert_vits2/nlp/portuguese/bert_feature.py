import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from style_bert_vits2.constants import Languages, DEFAULT_BERT_MODEL_PATHS
from style_bert_vits2.nlp.bert_models import load_model, load_tokenizer

def extract_bert_feature(
    text: str,
    word2ph: list[int],
    device: str,
    assist_text: str = None,
    assist_text_weight: float = 0.7,
) -> torch.Tensor:
    
    model_id = DEFAULT_BERT_MODEL_PATHS[Languages.PT]
    tokenizer = load_tokenizer(Languages.PT, model_id)
    model = load_model(Languages.PT, model_id)
    model.to(device)
    
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        
    # Usar a penúltima camada oculta como embedding (padrão BERT)
    # Shape: [batch_size, seq_len, hidden_size]
    res = outputs.hidden_states[-2]
    
    # Ajustar para corresponder aos fonemas (word2ph)
    # Esta é uma simplificação; idealmente faríamos alinhamento preciso
    # Mas como o BERT é token-based e VITS é phoneme-based, precisamos expandir
    
    # Nota: A implementação exata depende de como o tokenizer do BERT PT divide as palavras
    # e como isso se alinha com os fonemas.
    # Para simplificar, vamos interpolar para o tamanho dos fonemas
    
    res = res.squeeze(0).transpose(0, 1) # [hidden_size, seq_len]
    
    # Expandir para o número total de fonemas
    total_phones = sum(word2ph)
    res = torch.nn.functional.interpolate(
        res.unsqueeze(0), 
        size=total_phones, 
        mode='linear', 
        align_corners=True
    ).squeeze(0) # [hidden_size, total_phones]

    return res
