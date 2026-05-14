"""
Main execution script for CMPE 346 Final Project - NER.
Orchestrates data preprocessing, model training, and evaluation.
"""

from preprocessing import DataPreprocessor
from train_model import ModelTrainer
from evaluate import ModelEvaluator
from config import DATASETS, MODELS

def main():
    print("Starting CMPE 346 Final Project Workflow: Named Entity Recognition")
    
    # 1. Load and preprocess 3 distinct NER datasets.
    dataset_names = list(DATASETS.values())
    preprocessor = DataPreprocessor(dataset_names)
    preprocessor.load_datasets()
    datasets = preprocessor.preprocess()
    
    # 2. Initialize 5 different deep learning models for token classification.
    model_names = MODELS
    
    # 3. Fine-tune each of the 5 models on all 3 datasets independently.
    trainer = ModelTrainer(model_names, datasets)
    trainer.initialize_models()
    trained_models = trainer.train_all()
    
    # 4. Evaluate all model-dataset combinations to extract Accuracy and F1 scores.
    evaluator = ModelEvaluator(trained_models, datasets)
    evaluator.evaluate_all()
    
    # 5. Compile the evaluation metrics into a standardized comparative table.
    evaluator.compile_results_table()
    
    print("\nWorkflow completed successfully.")

if __name__ == "__main__":
    main()
