import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load model & tokenizer
model_name = "distilbert-base-german-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Load CSV
df = pd.read_csv("train.csv")
dataset = Dataset.from_pandas(df).train_test_split(test_size=0.2)

# Tokenize
def preprocess(examples):
    result = tokenizer(
        examples["Article"],
        examples["Comment"],
        padding=True,
        truncation=True,
        max_length=512
    )
    result["labels"] = examples["Relevance"]
    return result

tokenized = dataset.map(preprocess, batched=True)

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# Training arguments
args = TrainingArguments(
    output_dir="./results",
    # evaluation_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=15,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer.train()

# Save model
trainer.save_model("finetuned-distilbert")
tokenizer.save_pretrained("finetuned-distilbert")

# Optional: Save logits for calibration
outputs = trainer.predict(tokenized["test"])
np.save("logits.npy", outputs.predictions)
np.save("labels.npy", outputs.label_ids)
