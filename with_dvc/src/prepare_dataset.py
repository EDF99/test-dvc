from datasets import load_dataset
from transformers import MT5Tokenizer

def modify_newline_char(row_data):
    row_data["frase"] = row_data["frase"].replace("\n", "|")
    return row_data

def tokenize_function(row_data):
    model_inputs = tokenizer(row_data["frase"], max_length=512, truncation=True)
    labels = tokenizer(row_data["respuesta"], max_length=100, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

modelo_base_path = "../models/base-spa-mt5"
tokenizer = MT5Tokenizer.from_pretrained(modelo_base_path, legacy=False)

chimi_dataset = load_dataset("csv", data_dir="../chimi_synthetic_dataset")
chimi_dataset = chimi_dataset.map(modify_newline_char)
chimi_dataset = chimi_dataset.map(tokenize_function, batched=True)