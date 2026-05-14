"""
Evaluation module for Sequence Labeling (NER).
Evaluates all model-dataset combinations to extract Accuracy and F1 scores.
Compiles the evaluation metrics into a standardized comparative table.
"""
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, Trainer

class ModelEvaluator:
    def __init__(self, trained_models_paths, processed_datasets):
        self.trained_models_paths = trained_models_paths
        self.processed_datasets = processed_datasets
        self.results = []

    def evaluate_all(self):
        """Evaluates all model-dataset combinations to extract Accuracy and F1 scores."""
        for dataset_name, models_dict in self.trained_models_paths.items():
            for model_name, model_path in models_dict.items():
                print(f"\nEvaluating {model_name} on {dataset_name}...")
                
                # Load the fine-tuned model
                try:
                    model = AutoModelForTokenClassification.from_pretrained(model_path)
                except Exception as e:
                    print(f"Skipping evaluation for {model_name} on {dataset_name} due to missing weights. Error: {e}")
                    continue

                # We need the tokenizer to recreate the collator for evaluation
                # Since we already have the tokenized dataset, we just need ANY tokenizer that works with this model's inputs
                # In a real scenario, we'd load the tokenizer saved with the model.
                tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
                data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

                # Get the test dataset specifically for this model and dataset combination
                eval_dataset = self.processed_datasets[dataset_name][model_name]['test']

                # Use Hugging Face Trainer just for prediction/evaluation
                trainer = Trainer(
                    model=model,
                    eval_dataset=eval_dataset,
                    data_collator=data_collator
                )
                
                # Make predictions
                predictions, labels, _ = trainer.predict(eval_dataset)
                
                # Predictions are logits, we need the argmax to get the predicted class IDs
                predictions = np.argmax(predictions, axis=2)

                # Remove ignored index (special tokens)
                true_predictions = [
                    [p for (p, l) in zip(prediction, label) if l != -100]
                    for prediction, label in zip(predictions, labels)
                ]
                true_labels = [
                    [l for (p, l) in zip(prediction, label) if l != -100]
                    for prediction, label in zip(predictions, labels)
                ]
                
                # Flatten the lists for sklearn metrics
                flat_true_labels = [item for sublist in true_labels for item in sublist]
                flat_predictions = [item for sublist in true_predictions for item in sublist]
                
                # Calculate metrics
                accuracy = accuracy_score(flat_true_labels, flat_predictions)
                f1 = f1_score(flat_true_labels, flat_predictions, average='weighted', zero_division=0)
                
                self.results.append({
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "Accuracy": round(accuracy, 4),
                    "F1 Score": round(f1, 4)
                })

        return self.results

    def compile_results_table(self):
        """Compiles the evaluation metrics into a standardized comparative table."""
        if not self.results:
            print("No results to compile. Run evaluate_all() first.")
            return None
        
        df = pd.DataFrame(self.results)
        print("\n--- Evaluation Results ---")
        print(df.to_string(index=False))
        return df
