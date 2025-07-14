from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Load the fine-tuned model and tokenizer
model_dir = "finetuned-distilbert"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

# Set the model to evaluation mode (disable dropout, etc.)
model.eval()

# Example new text pair
Article = "This is the first example sentence."
Comment = "This sentence might be related to the first one."

# Encode the text pair
inputs = tokenizer(Article, Comment, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

# Disable gradient calculation (not needed for inference)
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Compute predicted class (0 or 1)
predicted_class_id = torch.argmax(logits, dim=1).item()

# Output the result
if predicted_class_id == 1:
    print("Prediction: The text pair is relevant")
else:
    print("Prediction: The text pair is not relevant")
