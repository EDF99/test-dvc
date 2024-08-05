from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, MT5ForConditionalGeneration, DataCollatorForSeq2Seq
from transformers.optimization import Adafactor
from common import model_tokenizer, compute_metrics_with_csv_building

modelo_base_path = "../models/base-spa-mt5"

base_model = MT5ForConditionalGeneration.from_pretrained(modelo_base_path)
base_model.config.max_length = 100

data_collator = DataCollatorForSeq2Seq(tokenizer=model_tokenizer, model=base_model)

batch_size = 8
training_args = Seq2SeqTrainingArguments(
    output_dir="../models/chimi-text-model",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    include_inputs_for_metrics=True,
    predict_with_generate=True,
    warmup_steps=20,
    save_total_limit=1,
    num_train_epochs=1,
    metric_for_best_model="exact_match",
    load_best_model_at_end=True
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

trainer = Seq2SeqTrainer(
    model=base_model,
    args=training_args,
    train_dataset=tokenized_chimi_dataset["train"],
    eval_dataset=tokenized_chimi_dataset["validation"],
    data_collator=data_collator,
    tokenizer=model_tokenizer,
    optimizers=(optimizer, None),
    compute_metrics=compute_metrics_with_csv_building()
)
trainer.train()