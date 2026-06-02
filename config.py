"""
Configuration file for the CMPE 346 Final Project.
Contains Kaggle dataset links, Hugging Face model configurations, and training parameters.
"""

# Kaggle Dataset Paths (format: "username/dataset-name")
DATASETS = {
    "dataset_1_entity_annotated": "namanj27/ner-dataset", 
    "dataset_2_conll2003": "juliangong/conll2003",
    "dataset_3_ner_corpus": "naseralqaydeh/named-entity-recognition-ner-corpus"
}

# Hugging Face Pre-trained Models (2 models for Colab training)
MODELS = [
    "bert-base-cased",
    "roberta-base",
    "distilbert-base-cased",
]

# Training Parameters (realistic hyperparameters for NER)
TRAINING_ARGS = {
    "learning_rate": 3e-5,
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 32,
    "num_train_epochs": 4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
}

# General Configuration
RANDOM_SEED = 42
MAX_LENGTH = 128
DEMO_MODE = False # Set to False for full training
