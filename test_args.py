import sys
sys.path.insert(0, "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages")

from transformers import TrainingArguments
print(TrainingArguments.__module__)


# from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    save_total_limit=1,
)

print("✅ OK! TrainingArguments created:")
print(args)
