import yaml
import argparse
from datasets import load_dataset
from transformers import MT5Tokenizer

def modify_newline_char(row_data):
    row_data["frase"] = row_data["frase"].replace("\n", "|")
    return row_data

def tokenize_function(row_data, tokenizer):
    model_inputs = tokenizer(row_data["frase"], max_length=512, truncation=True)
    labels = tokenizer(row_data["respuesta"], max_length=100, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def preprocess_dataset(config_path: str) -> None:
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    chimi_dataset = load_dataset("csv", data_files=config["data"]["raw_dataset"])
    print(chimi_dataset)
    chimi_dataset = chimi_dataset.map(modify_newline_char)
    tokenizer = MT5Tokenizer.from_pretrained(config["base"]["base_model_path"], legacy=False)
    chimi_dataset = chimi_dataset.map(tokenize_function, batched=True, fn_kwargs={'tokenizer': tokenizer})
    chimi_train_test = chimi_dataset["train"].train_test_split(test_size=config["data"]["validation_and_test_size"], seed=config["base"]["seed"])
    print(chimi_train_test)
    chimi_train_validation = chimi_train_test["train"].train_test_split(test_size=chimi_train_test["test"].num_rows, seed=config["base"]["seed"])

    chimi_dataset["train"] = chimi_train_validation["train"]
    chimi_dataset["validation"] = chimi_train_validation["test"]
    chimi_dataset["test"] = chimi_train_test["test"]
    print(chimi_dataset)
    chimi_dataset.save_to_disk(config["data"]["preprocessed_dataset"])

    # chimi_dataset["train"].save_to_disk(config["data"]["preprocessed_dataset"])
    
    print("Preprocessing data complete...")
    print("Dataset splitted into train, validation and test ...")
    print(f'Train: {chimi_dataset["train"].num_rows} rows || Validation: {chimi_dataset["validation"].num_rows} rows || Test: {chimi_dataset["test"].num_rows} rows')

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()
    preprocess_dataset(config_path=args.config)