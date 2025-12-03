import re
from num2words import num2words

# Abreviações comuns
_abbreviations = [
    (re.compile(r'\bSr\.'), 'Senhor'),
    (re.compile(r'\bSra\.'), 'Senhora'),
    (re.compile(r'\bDr\.'), 'Doutor'),
    (re.compile(r'\bDra\.'), 'Doutora'),
    (re.compile(r'\bProf\.'), 'Professor'),
    (re.compile(r'\bProfa\.'), 'Professora'),
    (re.compile(r'\bEng\.'), 'Engenheiro'),
    (re.compile(r'\bEnga\.'), 'Engenheira'),
    (re.compile(r'\bAv\.'), 'Avenida'),
    (re.compile(r'\bR\.'), 'Rua'),
    (re.compile(r'\bn\.º'), 'número'),
    (re.compile(r'\bpág\.'), 'página'),
    (re.compile(r'\betc\.'), 'etcetera'),
    (re.compile(r'\bV\. Exa\.'), 'Vossa Excelência'),
]

def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text

def expand_numbers(text):
    return re.sub(r'\d+', lambda x: num2words(int(x.group(0)), lang='pt_BR'), text)

def expand_currency(text):
    # R$ 10,00 -> dez reais
    # Simplificação: apenas remove o símbolo e trata como número, idealmente seria mais complexo
    # Mas num2words tem suporte a currency? Tem to_currency mas precisa de float.
    # Vamos fazer algo simples para R$, $, €
    
    # R$
    text = re.sub(r'R\$\s*(\d+)', lambda x: num2words(int(x.group(1)), lang='pt_BR') + ' reais', text)
    # $
    text = re.sub(r'\$\s*(\d+)', lambda x: num2words(int(x.group(1)), lang='pt_BR') + ' dólares', text)
    # €
    text = re.sub(r'€\s*(\d+)', lambda x: num2words(int(x.group(1)), lang='pt_BR') + ' euros', text)
    
    return text

def collapse_whitespace(text):
    return re.sub(r'\s+', ' ', text)

def normalize_text(text: str) -> str:
    """
    Normaliza o texto em português (portuguese_cleaners):
    1. Expande abreviações
    2. Expande moedas
    3. Expande números
    4. Converte para minúsculas
    5. Remove espaços extras
    """
    text = expand_abbreviations(text)
    text = expand_currency(text)
    text = expand_numbers(text)
    text = collapse_whitespace(text)
    # text = text.lower() # Opcional: phonemizer geralmente lida bem com case, mas VITS costuma usar lower.
    # Vamos manter o case original por enquanto pois o BERT é cased.
    # O BERT 'neuralmind/bert-base-portuguese-cased' É CASED. Então NÃO devemos dar lower.
    
    return text.strip()

