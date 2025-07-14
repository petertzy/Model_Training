from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import EarlyStoppingCallback

model_path = "finetuned-distilbert"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

df = pd.read_csv("new_train.csv")
dataset = Dataset.from_pandas(df).train_test_split(test_size=0.2)

def preprocess(examples):
    result = tokenizer(
        examples["Article"],
        examples["Comment"],
        padding="max_length",
        truncation=True,
        max_length=512
    )
    result["labels"] = examples["Relevance"]
    return result

tokenized = dataset.map(preprocess, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    # evaluation_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch", 
    num_train_epochs=15,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    seed=42,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=8)]
)

trainer.train()
trainer.save_model("finetuned-distilbert-continued")
tokenizer.save_pretrained("finetuned-distilbert-continued")
