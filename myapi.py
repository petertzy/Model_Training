from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

model_path = "finetuned-distilbert"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

class TextPair(BaseModel):
    Article: str
    Comment: str

@app.post("/predict")
def predict(data: TextPair):
    inputs = tokenizer(data.Article, data.Comment, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits[0]
    probs = torch.softmax(logits, dim=0)
    pred_label = torch.argmax(probs).item()

    logits_list = logits.detach().cpu().tolist()

    return {
        "Relevance": pred_label,
        "Confidence": probs[pred_label].item(),
        "Logits": logits_list
    }
