# Colab Running Guide

This guide assumes the project folder is named `cmpe346-pii-sequence-labeling`.

## 1. Open Colab With GPU

Runtime > Change runtime type > T4 GPU.

Then mount Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Go to the project folder:

```python
%cd /content/drive/MyDrive/cmpe346-pii-sequence-labeling
```

## 2. Install Libraries

```python
!pip install -r requirements.txt
```

Restart runtime if Colab asks for it, then run the mount and `%cd` cells again.

## 3. Download Datasets

Option A: Download manually from Kaggle and upload/extract into:

```text
data/raw/ai4privacy/
data/raw/pii_external/
data/raw/cleaned_pii/
```

Option B: Use Kaggle API:

```python
!mkdir -p ~/.kaggle
# Upload kaggle.json from your Kaggle account before this cell.
!cp kaggle.json ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json

!mkdir -p data/raw/ai4privacy data/raw/pii_external data/raw/cleaned_pii
!kaggle datasets download -d verracodeguacas/ai4privacy-pii -p data/raw/ai4privacy --unzip
!kaggle datasets download -d alejopaullier/pii-external-dataset -p data/raw/pii_external --unzip
!kaggle datasets download -d langdonholmes/cleaned-repository-of-annotated-pii -p data/raw/cleaned_pii --unzip
```

## 4. Convert Datasets to BIO

```python
!python src/prepare_dataset.py --input_dir data/raw/ai4privacy --output_path data/processed/ai4privacy.jsonl --dataset_type auto
!python src/prepare_dataset.py --input_dir data/raw/pii_external --output_path data/processed/pii_external.jsonl --dataset_type auto
!python src/prepare_dataset.py --input_dir data/raw/cleaned_pii --output_path data/processed/cleaned_pii.jsonl --dataset_type auto
```

Each command also writes a `.summary.json` file. Open it to check record count and label names.

## 5. First Smoke Test

Always start small:

```python
!python src/train_transformer_token.py \
  --model_name distilbert-base-cased \
  --dataset_path data/processed/ai4privacy.jsonl \
  --output_dir outputs/smoke_distilbert_ai4privacy \
  --max_samples 1000 \
  --num_train_epochs 1
```

If this works, the preprocessing and evaluation pipeline is healthy.

## 6. Main Experiments

Run these for each dataset by changing `--dataset_path` and `--output_dir`.

LSTM:

```python
!python src/train_lstm.py \
  --dataset_path data/processed/ai4privacy.jsonl \
  --output_dir outputs/lstm_ai4privacy \
  --num_train_epochs 8
```

BERT:

```python
!python src/train_transformer_token.py \
  --model_name bert-base-cased \
  --dataset_path data/processed/ai4privacy.jsonl \
  --output_dir outputs/bert_ai4privacy \
  --num_train_epochs 3
```

DistilBERT:

```python
!python src/train_transformer_token.py \
  --model_name distilbert-base-cased \
  --dataset_path data/processed/ai4privacy.jsonl \
  --output_dir outputs/distilbert_ai4privacy \
  --num_train_epochs 3
```

T5:

```python
!python src/train_seq2seq.py \
  --model_name t5-small \
  --dataset_path data/processed/ai4privacy.jsonl \
  --output_dir outputs/t5_ai4privacy \
  --max_samples 5000 \
  --num_train_epochs 2
```

mBART:

```python
!python src/train_seq2seq.py \
  --model_name facebook/mbart-large-50 \
  --dataset_path data/processed/ai4privacy.jsonl \
  --output_dir outputs/mbart_ai4privacy \
  --max_samples 2000 \
  --num_train_epochs 1 \
  --batch_size 1
```

## 7. Result Table

For the report, collect `test_metrics.json` files from `outputs/`.

Suggested table:

```text
Dataset | Model | Precision | Recall | F1 | Accuracy
```

For LSTM, BERT, and DistilBERT, prioritize F1 from `seqeval`.

For T5 and mBART, the script reports ROUGE-L because they generate entity text instead of token labels. In the report, explicitly say that encoder-only models were evaluated with sequence-labeling F1, while seq2seq models were evaluated with generated entity overlap.

## Why We Do It This Way

The datasets may store annotations differently. Some have token-level BIO labels, while others store text spans or privacy masks. The preparation script normalizes all of them into the same token/label format so LSTM, BERT, and DistilBERT can be compared fairly.

T5 and mBART are not natural token-classification models. They are encoder-decoder models, so we formulate the same PII task as text generation: input text goes in, and the model generates a list of PII entities. This keeps the five-model requirement while respecting the architecture.

