from transformers import AutoTokenizer, AutoModelForSequenceClassification

# model_name = "distilbert-base-uncased"
model_name = "distilbert-base-german-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)