import re
from unidecode import unidecode

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

def lowercase(text):
  return text.lower()

def collapse_whitespace(text):
  return re.sub(_whitespace_re, ' ', text)

def normalize_text(text):
  '''Pipeline for Portuguese text'''
  # Accents and special characters are important in Portuguese, so we don't convert to ascii
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text
