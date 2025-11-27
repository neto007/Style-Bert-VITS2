import re
from typing import List, Tuple

from phonemizer.backend import EspeakBackend

from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.nlp.symbols import PUNCTUATIONS, SYMBOLS

# Initialize phonemizer
backend = EspeakBackend(language='pt-br', preserve_punctuation=True, with_stress=True)

def g2p(text: str) -> Tuple[List[str], List[int], List[int]]:
    phones = []
    tones = []
    phone_len = []
    
    # BERT tokenizer for word alignment
    words = __text_to_words(text)

    for word in words:
        temp_phones = []
        temp_tones = []
        
        # Join subwords if any (though __text_to_words should handle this)
        full_word = "".join(word)
        
        if full_word in PUNCTUATIONS:
            temp_phones.append(full_word)
            temp_tones.append(0)
        else:
            phoneme_str = backend.phonemize([full_word], strip=True)[0]
            tokens = phoneme_str.split()
            clean_phonemes = [p for p in tokens if p in SYMBOLS]
            
            for p in clean_phonemes:
                temp_phones.append(p)
                temp_tones.append(0) # No tones in PT for now

        phones += temp_phones
        tones += temp_tones
        phone_len.append(len(temp_phones))

    word2ph = []
    for token, pl in zip(words, phone_len):
        word_len = len(token)
        word2ph += __distribute_phone(pl, word_len)

    phones = ["_"] + phones + ["_"]
    tones = [0] + tones + [0]
    word2ph = [1] + word2ph + [1]
    
    assert len(phones) == len(tones), text
    assert len(phones) == sum(word2ph), text

    return phones, tones, word2ph

def __distribute_phone(n_phone: int, n_word: int) -> List[int]:
    phones_per_word = [0] * n_word
    for task in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1
    return phones_per_word

def __text_to_words(text: str) -> List[List[str]]:
    tokenizer = bert_models.load_tokenizer(Languages.PT)
    tokens = tokenizer.tokenize(text)
    words = []
    for idx, t in enumerate(tokens):
        if t.startswith("##"):
            if words:
                words[-1].append(t[2:])
            else:
                # Should not happen if text is valid, but handle gracefully
                words.append([t[2:]])
        elif t in PUNCTUATIONS:
            words.append([t])
        else:
            words.append([t])
    return words
