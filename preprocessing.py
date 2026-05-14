"""
Preprocessing module for Sequence Labeling (NER).
Loads and preprocesses NER datasets from Kaggle into Hugging Face Dataset format.
"""
import os
import subprocess
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

load_dotenv()

class DataPreprocessor:
    def __init__(self, dataset_names):
        self.dataset_names = dataset_names
        self.raw_data = {}
        self.download_dir = "data"

    def load_datasets(self):
        """Loads Kaggle datasets and parses them into Pandas DataFrames."""
        os.makedirs(self.download_dir, exist_ok=True)

        for name in self.dataset_names:
            try:
                dataset_path = os.path.join(self.download_dir, name.split('/')[-1])
                if not os.path.exists(dataset_path):
                     print(f"Downloading {name}...")
                     subprocess.run(["kaggle", "datasets", "download", "-d", name, "-p", dataset_path, "--unzip"], check=True)
                
                # Parse datasets based on their known structure
                if "ner-dataset" in name:
                    # Namanj27 NER Dataset
                    csv_path = os.path.join(dataset_path, "ner_datasetreference.csv")
                    if os.path.exists(csv_path):
                        df = pd.read_csv(csv_path, encoding="latin1")
                        df = df.ffill() # Forward fill 'Sentence: 1'
                        
                        # Group by sentence
                        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                                          s["POS"].values.tolist(),
                                                                          s["Tag"].values.tolist())]
                        grouped = df.groupby("Sentence #").apply(agg_func).reset_index(drop=True)
                        
                        sentences = [[s[0] for s in group] for group in grouped]
                        tags = [[s[2] for s in group] for group in grouped]
                        
                        self.raw_data[name] = {"tokens": sentences, "ner_tags": tags}
                        print(f"Parsed {name}: {len(sentences)} sentences.")
                
                elif "named-entity-recognition-ner-corpus" in name:
                    # Naseralqaydeh NER Corpus
                    csv_path = os.path.join(dataset_path, "ner.csv")
                    if os.path.exists(csv_path):
                        df = pd.read_csv(csv_path, encoding="latin1")
                        df = df.ffill()
                        
                        # Print columns to debug if needed
                        # print(f"Columns for {name}: {df.columns.tolist()}")

                        # This dataset has slightly different column names, usually Sentence #, Word, Tag
                        sentence_col = next((col for col in df.columns if "sentence" in col.lower()), "Sentence #")
                        word_col = next((col for col in df.columns if "word" in col.lower() or "token" in col.lower()), "Word")
                        tag_col = next((col for col in df.columns if "tag" in col.lower() or "label" in col.lower()), "Tag")

                        try:
                            agg_func = lambda s: [(w, t) for w, t in zip(s[word_col].values.tolist(),
                                                                         s[tag_col].values.tolist())]
                            grouped = df.groupby(sentence_col).apply(agg_func).reset_index(drop=True)
                            
                            sentences = [[str(s[0]) for s in group] for group in grouped]
                            tags = [[str(s[1]) for s in group] for group in grouped]
                            
                            self.raw_data[name] = {"tokens": sentences, "ner_tags": tags}
                            print(f"Parsed {name}: {len(sentences)} sentences.")
                        except KeyError as e:
                             print(f"KeyError in parsing {name}. Columns found: {df.columns.tolist()}. Error: {e}")

            except Exception as e:
                print(f"Error processing dataset {name}: {e}")

    def get_unique_labels(self):
        """Extracts unique NER tags across all loaded datasets."""
        unique_tags = set()
        for data in self.raw_data.values():
            for tag_list in data["ner_tags"]:
                unique_tags.update(tag_list)
        
        unique_tags = sorted(list(unique_tags))
        
        # Ensure 'O' is always at index 0 for convention
        if "O" in unique_tags:
            unique_tags.remove("O")
            unique_tags.insert(0, "O")
            
        label2id = {tag: id for id, tag in enumerate(unique_tags)}
        id2label = {id: tag for id, tag in enumerate(unique_tags)}
        
        return unique_tags, label2id, id2label

    def align_labels_with_tokens(self, labels, word_ids):
        """Aligns labels with sub-word tokens generated by the tokenizer."""
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                # Start of a new word!
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                # Special token
                new_labels.append(-100)
            else:
                # Same word as previous token (sub-word token)
                label = labels[word_id]
                # If the original label is B-xxx, we change it to I-xxx for sub-words
                # Since our tags are strings here, we handle this after numerical mapping, 
                # but for simplicity, we assign -100 to subwords in basic setups
                new_labels.append(-100) 
                
        return new_labels

    def preprocess(self, model_names):
        """Converts raw data to HF DatasetDicts and tokenizes them for each model."""
        unique_tags, label2id, id2label = self.get_unique_labels()
        print(f"\nDiscovered {len(unique_tags)} unique labels: {unique_tags}")
        
        tokenizers = {}
        for model_name in model_names:
            print(f"Loading tokenizer for {model_name}...")
            tokenizers[model_name] = AutoTokenizer.from_pretrained(
                model_name, 
                token=os.environ.get("HF_TOKEN"),
                add_prefix_space=True # Helps with some RoBERTa/Albert variants
            )

        processed_datasets = {}
        
        for dataset_name, data in self.raw_data.items():
            # Create a base Hugging Face Dataset
            hf_dataset = Dataset.from_dict({
                "tokens": data["tokens"],
                "ner_tags_str": data["ner_tags"] # Keep strings initially
            })
            
            # Map string tags to IDs
            def map_tags_to_ids(example):
                example["ner_tags"] = [label2id[tag] for tag in example["ner_tags_str"]]
                return example
            hf_dataset = hf_dataset.map(map_tags_to_ids)
            
            # Simple train/test split (80/20)
            dataset_dict = hf_dataset.train_test_split(test_size=0.2, seed=42)
            
            # Tokenize for each model separately, because each tokenizer has a different vocabulary
            processed_datasets[dataset_name] = {}
            for model_name, tokenizer in tokenizers.items():
                
                def tokenize_and_align(examples):
                    tokenized_inputs = tokenizer(
                        examples["tokens"], 
                        truncation=True, 
                        is_split_into_words=True,
                        max_length=128
                    )
                    
                    labels = []
                    for i, label in enumerate(examples["ner_tags"]):
                        word_ids = tokenized_inputs.word_ids(batch_index=i)
                        labels.append(self.align_labels_with_tokens(label, word_ids))

                    tokenized_inputs["labels"] = labels
                    return tokenized_inputs

                print(f"Tokenizing {dataset_name} for {model_name}...")
                tokenized_dataset = dataset_dict.map(tokenize_and_align, batched=True, remove_columns=["tokens", "ner_tags_str", "ner_tags"])
                processed_datasets[dataset_name][model_name] = tokenized_dataset

        return processed_datasets, tokenizers, len(unique_tags), label2id, id2label
