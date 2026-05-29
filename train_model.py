"""
Training module for Sequence Labeling (NER).
Initializes 5 different deep learning models for token classification from Hugging Face.
Fine-tunes each of the 5 models on all datasets independently.
"""

import os
import torch
from dotenv import load_dotenv
from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from config import TRAINING_ARGS, RANDOM_SEED

# Load environment variables
load_dotenv()

class ModelTrainer:
    def __init__(self, model_names, datasets, num_labels, label2id, id2label):
        self.model_names = model_names
        self.datasets = datasets # Now expects {dataset_name: {model_name: DatasetDict}}
        self.models = {}
        self.num_labels = num_labels
        self.label2id = label2id
        self.id2label = id2label
        
        self.hf_token = os.environ.get("HF_TOKEN")

    def initialize_models(self):
        """Initializes deep learning models for token classification."""
        for name in self.model_names:
            print(f"Initializing model: {name}...")
            
            # Load the pre-trained model and adjust the classification head
            model = AutoModelForTokenClassification.from_pretrained(
                name, 
                num_labels=self.num_labels,
                label2id=self.label2id,
                id2label=self.id2label,
                token=self.hf_token,
                ignore_mismatched_sizes=True # Useful if pre-trained model has different num labels
            )
            self.models[name] = model

    def train_all(self, tokenizers):
        """Fine-tunes each model on each dataset."""
        trained_models = {}
        
        for dataset_name, models_dict in self.datasets.items():
            trained_models[dataset_name] = {}
            
            for model_name, dataset_dict in models_dict.items():
                model = self.models[model_name]
                print(f"\n{'='*50}\nTraining {model_name} on {dataset_name}\n{'='*50}")
                
                tokenizer = tokenizers[model_name]
                data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

                # Set up training arguments
                output_dir = f"./model_weights/{dataset_name.split('/')[-1]}_{model_name.replace('/', '-')}"
                
                training_args = TrainingArguments(
                    output_dir=output_dir,
                    learning_rate=TRAINING_ARGS["learning_rate"],
                    per_device_train_batch_size=TRAINING_ARGS["per_device_train_batch_size"],
                    per_device_eval_batch_size=TRAINING_ARGS["per_device_eval_batch_size"],
                    num_train_epochs=TRAINING_ARGS["num_train_epochs"],
                    weight_decay=TRAINING_ARGS["weight_decay"],
                    warmup_ratio=TRAINING_ARGS.get("warmup_ratio", 0.0),
                    eval_strategy=TRAINING_ARGS["eval_strategy"],
                    save_strategy=TRAINING_ARGS["save_strategy"],
                    load_best_model_at_end=TRAINING_ARGS["load_best_model_at_end"],
                    seed=RANDOM_SEED,
                    push_to_hub=False,
                )

                # Initialize Trainer
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=dataset_dict['train'],
                    eval_dataset=dataset_dict['test'], 
                    processing_class=tokenizer,
                    data_collator=data_collator,
                )

                # Execute fine-tuning
                trainer.train()
                
                # Save the final model
                trainer.save_model(output_dir)
                trained_models[dataset_name][model_name] = output_dir # Store path to trained model

        return trained_models
