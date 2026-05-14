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
    
    # Filter datasets to only those that successfully downloaded and have parsers
    dataset_names = [DATASETS["dataset_1_entity_annotated"], DATASETS["dataset_3_ner_corpus"]]
    
    # 1. Load and preprocess datasets.
    preprocessor = DataPreprocessor(dataset_names)
    preprocessor.load_datasets()
    
    # Convert Pandas DataFrames into Tokenized Hugging Face DatasetDicts
    processed_datasets, tokenizers, num_labels, label2id, id2label = preprocessor.preprocess(MODELS)
    
    if not processed_datasets:
        print("ERROR: No datasets were successfully preprocessed. Exiting.")
        return

    # 2 & 3. Initialize and Fine-tune each of the models on all datasets.
    # We restructuring slightly: since each dataset is tokenized differently for each model,
    # we need to pass this nested structure to the trainer.
    
    trainer = ModelTrainer(
        model_names=MODELS, 
        datasets=processed_datasets, # Note: This is now a nested dict: {dataset: {model: HF_Dataset}}
        num_labels=num_labels, 
        label2id=label2id, 
        id2label=id2label
    )
    
    trainer.initialize_models()
    trained_models_paths = trainer.train_all(tokenizers)
    
    # 4. Evaluate all model-dataset combinations.
    # Note: Evaluator will need to be updated to use trainer.predict() in the future,
    # but the paths are now passed successfully.
    evaluator = ModelEvaluator(trained_models_paths, processed_datasets)
    evaluator.evaluate_all()
    
    # 5. Compile the evaluation metrics into a standardized comparative table.
    evaluator.compile_results_table()
    
    print("\nWorkflow completed successfully.")

if __name__ == "__main__":
    main()
