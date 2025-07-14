from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Local model path, e.g. the directory where your fine-tuned model is saved
local_model_path = "./finetuned-distilbert"

# Load the model and tokenizer from the local path
model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# Upload to the Hugging Face Hub
model.push_to_hub("peter-haan/finetuned-distilbert")
tokenizer.push_to_hub("peter-haan/finetuned-distilbert")

print("Upload complete!")

