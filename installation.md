# From Scratch: Fine-tuning a Hugging Face Model for Text Pair Classification (Comment-Article Relevance)

---

## Contents

1. Introduction

2. Environment Setup

3. Installing Miniconda

4. Creating a Virtual Environment and Installing Dependencies

5. Preparing Training Data

6. Downloading and Loading a Pretrained Model

7. Fine-tuning the Model

8. Saving and Loading the Fine-tuned Model

9. Inference (Testing)

10. Further Suggestions

---

## 1. Introduction

This tutorial shows you how to fine-tune a Hugging Face Transformer model (e.g., DistilBERT) to perform text pair classification: determining if a comment is relevant to a given article.

Input: a pair of texts (article content and comment)

Output: relevance label (1 for relevant, 0 for not relevant)

---

## 2. Environment Setup

You need Python 3 installed. It’s recommended to use Miniconda for managing isolated environments.

---

## 3. Installing Miniconda

Download and install Miniconda based on your operating system:

* **Mac Apple Silicon (M1/M2):**

[https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.pkg](https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.pkg)

* **Mac Intel:**

[https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86\_64.pkg](https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.pkg)

* **Windows:**

[https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86\_64.exe](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe)

* **Linux:**

[https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86\_64.sh](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh)

After installation, open your terminal or command prompt.

---

## 4. Creating a Virtual Environment and Installing Dependencies

Open terminal and run:

```bash

conda create -n hf_env python=3.10 # run once to create environment

conda activate hf_env # activate environment

```

Then install required Python packages:

```bash

pip install torch transformers datasets scikit-learn

```

> **Note**

> For M1 Mac, it is recommended to install PyTorch with Metal backend support:

>

> ```bash

> pip install torch torchvision torchaudio

> ```

>

> See [PyTorch official instructions](https://pytorch.org/get-started/locally/#macos).

---

## 5. Preparing Training Data

Create a CSV file named `train.csv` with content like:

```csv

Article,Comment,label

"Apple releases new chip","The new laptop has a bigger battery",1

"Apple releases new chip","I love eating pizza",0

```

* `Article` is the article content

* `Comment` is the comment

* `label` is relevance (1 for relevant, 0 for not relevant)

---

## 6. Downloading and Loading a Pretrained Model

Create a Python script `load_model.py`:

```python

from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

print("Model and tokenizer loaded successfully!")

```

Run:

```bash

python load_model.py

```

This will download the model (\~250MB) and cache it locally.

---

## 7. Fine-tuning the Model

Create `train_model.py`:

```python

from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

import numpy as np

from sklearn.metrics import accuracy_score

# Load dataset

dataset = load_dataset("csv", data_files="train.csv")["train"].train_test_split(test_size=0.2)

# Load model and tokenizer

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenization function

def tokenize(batch):

return tokenizer(batch["Article"], batch["Comment"], padding="max_length", truncation=True, max_length=512)

tokenized = dataset.map(tokenize, batched=True)

# Training arguments

training_args = TrainingArguments(

output_dir="./results",

# evaluation_strategy="epoch",

num_train_epochs=3,

per_device_train_batch_size=4,

per_device_eval_batch_size=4,

save_total_limit=1,

)

# Metrics function

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

)

trainer.train()

# Save fine-tuned model and tokenizer

trainer.save_model("finetuned-distilbert")

tokenizer.save_pretrained("finetuned-distilbert")

```

Run:

```bash

python train_model.py

```

---

## 8. Saving and Loading the Fine-tuned Model

The fine-tuned model and tokenizer are saved in the folder `finetuned-distilbert`.

You can reload them in another script like:

```python

from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "finetuned-distilbert"

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForSequenceClassification.from_pretrained(model_path)

```

---

## 9. Inference (Testing)

Create `test_finetuned_model.py`:

```python

import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "finetuned-distilbert"

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForSequenceClassification.from_pretrained(model_path)

Article = "Apple releases new chip"

Comment = "The new laptop has a bigger battery"

inputs = tokenizer(Article, Comment, return_tensors="pt", truncation=True, padding=True)

outputs = model(**inputs)

probs = torch.softmax(outputs.logits, dim=1)

pred_label = torch.argmax(probs).item()

print("Is relevant:", "Yes" if pred_label == 1 else "No")

```

Run:

```bash

python test_finetuned_model.py

```

---

## 10. Further Suggestions

* More training data usually improves accuracy

* Tune hyperparameters like learning rate, batch size, and epochs

* For bigger models or larger data, consider using cloud GPUs

* Wrap inference into a web API using Flask, FastAPI, or Streamlit for deployment

---

# Summary

You now have a full guide to:

* Set up your local environment (Mac M1 or other)

* Download pretrained models from Hugging Face

* Prepare your training data

* Fine-tune a text pair classification model

* Save, load, and test your fine-tuned model
