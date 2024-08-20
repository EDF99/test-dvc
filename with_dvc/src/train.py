import yaml
import argparse
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, MT5ForConditionalGeneration, DataCollatorForSeq2Seq, MT5Tokenizer, GenerationConfig
from datasets import load_from_disk
from transformers.optimization import Adafactor
from common import compute_metrics_with_csv_building
from pynvml import *
from transformers.integrations import DVCLiveCallback
from dvclive import Live
import shutil
import os

os.environ["HF_DVCLIVE_LOG_MODEL"] = "true"

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_training_summary(result):
    print(f"\nTime: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()

def train_model(config_path: str) -> None:
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    chimi_dataset = load_from_disk(config["data"]["preprocessed_dataset"])
    base_model = MT5ForConditionalGeneration.from_pretrained(config["base"]["base_model_path"])
    tokenizer = MT5Tokenizer.from_pretrained(config["base"]["base_model_path"], legacy=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=base_model)

    generation_config = GenerationConfig.from_model_config(base_model.config)
    generation_config.max_length = 100
    base_model.generation_config = generation_config
    checkpoints_tmp_folder = "models/checkpoints"

    training_args = Seq2SeqTrainingArguments(
        output_dir=checkpoints_tmp_folder,
        overwrite_output_dir=True,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=config["train"]["batch_size"],
        per_device_eval_batch_size=config["train"]["batch_size"],
        include_inputs_for_metrics=True,
        predict_with_generate=True,
        warmup_steps=20,
        save_total_limit=1,
        num_train_epochs=config["train"]["epochs"],
        metric_for_best_model="exact_match",
        load_best_model_at_end=True,
        report_to="none"
    )

    optimizer = Adafactor(
        base_model.parameters(),
        lr=1e-3,
        clip_threshold=1.0,
        weight_decay=0.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
    )

    live = Live(config["train"]["reports_path"])
    compute_metrics_func = compute_metrics_with_csv_building(tokenizer)
    trainer = Seq2SeqTrainer(
        model=base_model,
        args=training_args,
        train_dataset=chimi_dataset["train"],
        eval_dataset=chimi_dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        optimizers=(optimizer, None),
        compute_metrics=compute_metrics_func,
        callbacks=[DVCLiveCallback(live=live)]
    )
    print("Training will start...\n")
    result = trainer.train()
    print_training_summary(result)
    trainer.save_model(config["train"]["trained_model_path"])

    if os.path.exists(checkpoints_tmp_folder):
        shutil.rmtree(checkpoints_tmp_folder)

    print("\nTraining has finished...")

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()
    train_model(config_path=args.config)