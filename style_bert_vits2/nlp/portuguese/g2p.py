from phonemizer.backend import EspeakBackend
from style_bert_vits2.nlp.symbols import PT_SYMBOLS, PUNCTUATION_SYMBOLS

# Initialize backend once
_backend = None

def get_backend():
    global _backend
    if _backend is None:
        _backend = EspeakBackend(
            language='pt-br',
            preserve_punctuation=True,
            with_stress=True
        )
    return _backend

def g2p(text: str) -> tuple[list[str], list[int], list[int]]:
    """
    Converte texto em fonemas usando phonemizer (espeak-ng backend)
    """
    # Converter para fonemas
    backend = get_backend()
    phones_str = backend.phonemize(
        [text],
        strip=True
    )[0]
    
    phones = []
    tones = []
    word2ph = []
    
    # Processar fonemas
    # Nota: Esta é uma implementação simplificada. 
    # Precisamos mapear os fonemas do espeak para nossos PT_SYMBOLS
    
    current_word_phones = 0
    
    for p in phones_str:
        if p in PT_SYMBOLS:
            phones.append(p)
            tones.append(0) # Tones não usados em PT da mesma forma que ZH
            current_word_phones += 1
        elif p in PUNCTUATION_SYMBOLS:
            phones.append(p)
            tones.append(0)
            word2ph.append(current_word_phones)
            word2ph.append(1) # Pontuação conta como 1
            current_word_phones = 0
        elif p == ' ':
            if current_word_phones > 0:
                word2ph.append(current_word_phones)
                current_word_phones = 0
        
    if current_word_phones > 0:
        word2ph.append(current_word_phones)
        
    return phones, tones, word2ph
