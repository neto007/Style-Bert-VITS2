import re
from num2words import num2words

def normalize_text(text: str) -> str:
    """
    Normaliza o texto em português:
    - Converte números para extenso
    - Remove caracteres indesejados
    """
    # Converter números para extenso
    text = re.sub(r"\d+", lambda x: num2words(int(x.group(0)), lang='pt_BR'), text)
    
    # Substituições comuns
    text = text.replace("sr.", "senhor").replace("dr.", "doutor").replace("sra.", "senhora")
    
    return text
