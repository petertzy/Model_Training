#  How to Deploy on Hugging Face Spaces?

---

## ① Sign up for a Hugging Face Account

Go to the official website [https://huggingface.co](https://huggingface.co) and register.

---

## ② Upload Your Model to Hugging Face

First, upload your trained model to the Hugging Face Hub. For example:

```
distilbert-finetuned-relevance
```

The uploaded files should include:

* pytorch\_model.bin
* config.json
* tokenizer\_config.json
* vocab.txt
* special\_tokens\_map.json
* training\_args.bin (optional)

Alternatively, you can upload directly using the `transformers` library’s `push_to_hub()` method:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("path/to/your/local/model")
tokenizer = AutoTokenizer.from_pretrained("path/to/your/local/model")

model.push_to_hub("your-username/your-model-name")
tokenizer.push_to_hub("your-username/your-model-name")
```

Once uploaded, Hugging Face will provide a model URL, for example:

```
https://huggingface.co/your-username/your-model-name
```

---

## ③ Create a Hugging Face Space

Go to the Hugging Face website:

* Click ➕ **New Space**
* Choose:

  * Gradio (the easiest)
  * or Streamlit
* Name your Space, e.g.:

  ```
  relevance-classifier-demo
  ```

---

## ④ Build a Gradio Interface

Inside your Space, create a new file called `app.py` and write code similar to the following:

```python
import gradio as gr
from transformers import pipeline

pipe = pipeline(
    "text-classification",
    model="your-username/your-model-name"
)

def predict(article, comment):
    # Concatenate article + comment, or process separately,
    # depending on how your training data was structured
    text = f"{article} [SEP] {comment}"
    output = pipe(text, truncation=True)[0]
    label = output["label"]
    score = output["score"]
    return f"Label: {label} | Confidence: {score:.3f}"

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Article"),
        gr.Textbox(label="Comment")
    ],
    outputs="text",
    title="Relevance Classifier (German)"
)

demo.launch()
```

---

## ⑤ requirements.txt

In your Space, create a `requirements.txt` file and include:

```
transformers
torch
gradio
```

---

## ⑥ Deploy

* Commit your code
* Hugging Face will automatically deploy your Space
* Once completed, you’ll get a web URL, for example:

  ```
  https://huggingface.co/spaces/your-username/relevance-classifier-demo
  ```

Anyone visiting that link will be able to enter an Article and Comment and see the model’s prediction results live.

---

##  Is It Free?

* **Spaces are free** → for low traffic and CPU usage (the free tier may go to sleep if unused)
* If you want faster performance or prevent sleeping → upgrade to a paid plan (not very expensive)

If you have small datasets and a lightweight model, CPU is usually sufficient, and the free plan can easily cover your testing needs.

---

#  Alternative: Inference Endpoint

If you prefer an API rather than a web interface:

* Hugging Face offers Inference Endpoints
* Deploy and get an API URL
* Perfect for integrating into production systems
* **Paid service** (charged by the hour)
