import yaml
import argparse
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, MT5ForConditionalGeneration, DataCollatorForSeq2Seq, MT5Tokenizer
from datasets import load_from_disk
from common import compute_metrics_with_csv_building
import json

def evaluate_trained_model(config_path: str) -> None:
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    chimi_dataset = load_from_disk(config["data"]["preprocessed_dataset"])
    trained_model = MT5ForConditionalGeneration.from_pretrained(config["train"]["trained_model_path"])
    trained_tokenizer = MT5Tokenizer.from_pretrained(config["train"]["trained_model_path"])
    data_collator = DataCollatorForSeq2Seq(tokenizer=trained_tokenizer, model=trained_model)
    evaluate_args = Seq2SeqTrainingArguments(
        output_dir="models/checkpoints",
        logging_strategy="epoch",
        eval_strategy="epoch",
        per_device_eval_batch_size=config["train"]["batch_size"],
        include_inputs_for_metrics=True,
        predict_with_generate=True,
        metric_for_best_model="exact_match"
    )
    compute_metrics_func = compute_metrics_with_csv_building(trained_tokenizer)
    evaluate_trainer = Seq2SeqTrainer(
        model=trained_model,
        args=evaluate_args,
        eval_dataset=chimi_dataset["test"],
        data_collator=data_collator,
        tokenizer=trained_tokenizer,
        compute_metrics=compute_metrics_func
    )
    result = evaluate_trainer.evaluate()
    print(result)
    with open(config["evaluate"]["metrics_path"], 'w') as fp:
        json.dump(result, fp)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()
    evaluate_trained_model(config_path=args.config)