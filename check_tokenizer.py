from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
text = "Olá, mundo! Isso é um teste."
tokens = tokenizer.tokenize(text)
print(tokens)

text2 = "Paralelepípedo"
tokens2 = tokenizer.tokenize(text2)
print(tokens2)
