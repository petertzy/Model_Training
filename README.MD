### Preparation:

**Download and install Miniconda:**

* For Mac (Apple Silicon):
  [https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.pkg](https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.pkg)

* For Mac (Intel):
  [https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86\_64.pkg](https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.pkg)

* For Windows:
  [https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86\_64.exe](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe)

* For Linux:
  [https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86\_64.sh](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh)


**Create a virtual environment** by running the following command:

   ```bash
   python -m venv ~/myenv
   ```

**Install the required packages:**

```bash
pip install -r requirements.txt
```

**Activate the virtual environment:**

   * On **macOS/Linux**, run:

     ```bash
     source ~/myenv/bin/activate
     ```

   * On **Windows (PowerShell)**, run:

     ```powershell
     .\myenv\Scripts\Activate.ps1
     ```

   * On **Windows (cmd.exe)**, run:

     ```cmd
     myenv\Scripts\activate.bat
     ```

**When done, deactivate the virtual environment:**

   ```bash
   deactivate
   ```


### Get Started:

Run the following command to download the model（Can be omitted）:

```bash
python load_model.py # (only for the first time)
```

Run the following command to train the model:

```bash
python train_model.py
```

Run the following command to test the fine-tuned model on new data（Can be omitted）:

```bash
python test_finetuned_model.py
```

### Deploy:

**Local:**

```bash
uvicorn myapi:app --host 0.0.0.0 --port 8000
```

**temperature scaling:（Can be omitted）**
```bash
python temperature_scaling.py
```

**batch test**
```bash
python batch_test.py
```

### Application:

**User call** (or use curl to call the online API：curl -X POST http://yourserver.com:8000/predict \)

```bash
curl -X POST http://0.0.0.0:8000/predict \
-H "Content-Type: application/json" \
-d '{"Article": "Apple veröffentlicht neuen Chip", "Comment": "Der neue Laptop hat einen größeren Akku"}'
```

Expected response example:

```json
{
  "Relevance":1,
  "Confidence":0.5046111345291138,
  "Logits":[0.005788873881101608,0.02423405833542347]
}
```

**Continue training the existing model**

```bash
python continue_train_model.py
```

**Upload Model to Hugging Face**
Login to Hugging Face from local machine:
```bash
huggingface-cli login
```
Upload your model:
```bash
python upload_model.py
```
---


### Other useful information:

**List models in the cache**

```bash
ls ~/.cache/huggingface/hub
```

**Delete all models**

```bash
rm -rf ~/.cache/huggingface/hub
```

**Delete a specific model**

```bash
rm -rf ~/.cache/huggingface/hub/models--gpt2
rm -rf ~/.cache/huggingface/hub/models--distilbert-base-uncased
```

**Save a model to a custom folder:**

```python
model.save_pretrained("./finetuned-distilbert")
tokenizer.save_pretrained("./finetuned-distilbert")
```

Then load it later like this:

```python
model = AutoModelForSequenceClassification.from_pretrained("./finetuned-distilbert")
tokenizer = AutoTokenizer.from_pretrained("./finetuned-distilbert")
```

This way, your model stays with your project and doesn’t rely on external caches.
