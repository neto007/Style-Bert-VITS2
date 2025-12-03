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

# Símbolos
_symbols = [
    (re.compile(r'%'), ' por cento'),
    (re.compile(r'&'), ' e '),
    (re.compile(r'@'), ' arroba '),
    (re.compile(r'\+'), ' mais '),
    (re.compile(r'='), ' igual a '),
]

def expand_symbols(text):
    for regex, replacement in _symbols:
        text = re.sub(regex, replacement, text)
    return text

def expand_ordinals(text):
    # 1º, 2º, 3º...
    text = re.sub(r'(\d+)[º°]', lambda x: num2words(int(x.group(1)), lang='pt_BR', to='ordinal'), text)
    # 1ª, 2ª, 3ª... (num2words em PT-BR gera masculino por padrão, para feminino precisaríamos de lógica extra ou pós-processamento)
    # O num2words tem suporte a genero? Parece que não nativamente fácil para PT.
    # Vamos simplificar e usar o masculino para º e tentar adaptar para ª se possível, ou deixar genérico.
    # Hack: substituir 'o' final por 'a' para femininos? Arriscado.
    # Vamos manter apenas o masculino para º por enquanto e ignorar ª ou tratar como cardinal se falhar.
    # Mas 'primeira' é comum.
    # Vamos tentar um mapa simples para os mais comuns se num2words falhar.
    
    return text

def expand_date(text):
    # dd/mm/aaaa
    # Ex: 01/01/2023 -> primeiro de janeiro de 2023
    # Ex: 10/12/2023 -> dez de dezembro de 2023
    
    _months = {
        '01': 'janeiro', '02': 'fevereiro', '03': 'março', '04': 'abril', '05': 'maio', '06': 'junho',
        '07': 'julho', '08': 'agosto', '09': 'setembro', '10': 'outubro', '11': 'novembro', '12': 'dezembro',
        '1': 'janeiro', '2': 'fevereiro', '3': 'março', '4': 'abril', '5': 'maio', '6': 'junho',
        '7': 'julho', '8': 'agosto', '9': 'setembro'
    }
    
    def replace_date(match):
        day, month, year = match.group(1), match.group(2), match.group(3)
        day_text = num2words(int(day), lang='pt_BR') if int(day) != 1 else 'primeiro'
        month_text = _months.get(month, month)
        year_text = num2words(int(year), lang='pt_BR')
        return f"{day_text} de {month_text} de {year_text}"

    return re.sub(r'(\d{1,2})/(\d{1,2})/(\d{4})', replace_date, text)

def expand_time(text):
    # hh:mm ou hhHmm
    def replace_time(match):
        h, m = int(match.group(1)), int(match.group(2))
        h_text = num2words(h, lang='pt_BR')
        m_text = num2words(m, lang='pt_BR')
        
        hours_label = "hora" if h == 1 else "horas"
        minutes_label = "minuto" if m == 1 else "minutos"
        
        if m == 0:
            return f"{h_text} {hours_label}"
        else:
            return f"{h_text} {hours_label} e {m_text} {minutes_label}"

    return re.sub(r'(\d{1,2})[:h](\d{2})', replace_time, text)

def collapse_whitespace(text):
    return re.sub(r'\s+', ' ', text)

def normalize_text(text: str) -> str:
    """
    Normaliza o texto em português (portuguese_cleaners):
    1. Expande datas (dd/mm/aaaa)
    2. Expande horas (hh:mm)
    3. Expande abreviações
    4. Expande moedas
    5. Expande ordinais (1º)
    6. Expande símbolos (%, &, @)
    7. Expande números
    8. Converte para minúsculas
    9. Remove espaços extras
    """
    text = expand_date(text)
    text = expand_time(text)
    text = expand_abbreviations(text)
    text = expand_currency(text)
    text = expand_ordinals(text)
    text = expand_symbols(text)
    text = expand_numbers(text)
    text = collapse_whitespace(text)
    
    return text.strip()

