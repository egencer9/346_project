"""
Evaluation module for Sequence Labeling (NER).
Evaluates all model-dataset combinations to extract Accuracy and F1 scores.
Compiles the evaluation metrics into a standardized comparative table.
"""
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

class ModelEvaluator:
    def __init__(self, trained_models, datasets):
        self.trained_models = trained_models
        self.datasets = datasets
        self.results = []

    def evaluate_all(self):
        """Evaluates all model-dataset combinations to extract Accuracy and F1 scores."""
        for dataset_name, models in self.trained_models.items():
            for model_name, model in models.items():
                print(f"Evaluating {model_name} on {dataset_name}")
                
                # TODO: Replace these mocks with actual predictions and references
                # For NER, you'll need to flatten your lists of labels before passing to sklearn
                # e.g., true_labels = [label for sequence in true_sequences for label in sequence]
                mock_true_labels = [0, 1, 2, 0, 1]
                mock_predictions = [0, 1, 2, 0, 0]
                
                accuracy = accuracy_score(mock_true_labels, mock_predictions)
                # average='macro' or 'weighted' is usually best for NER class imbalance
                f1 = f1_score(mock_true_labels, mock_predictions, average='weighted')
                
                self.results.append({
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "Accuracy": accuracy,
                    "F1 Score": f1
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
