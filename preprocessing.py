"""
Preprocessing module for Sequence Labeling (NER).
Loads and preprocesses 3 distinct NER datasets from Kaggle.
"""
import os
import subprocess
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class DataPreprocessor:
    def __init__(self, dataset_names):
        self.dataset_names = dataset_names
        self.datasets = {}
        self.download_dir = "data"

    def load_datasets(self):
        """Loads the 3 distinct NER datasets from Kaggle."""
        
        # Check if Kaggle credentials are set
        if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
             print("WARNING: Kaggle credentials not found in environment variables.")
             print("Make sure you have a .env file with KAGGLE_USERNAME and KAGGLE_KEY.")
             print("Or ensure ~/.kaggle/kaggle.json exists.")
        
        os.makedirs(self.download_dir, exist_ok=True)

        for name in self.dataset_names:
            print(f"Loading dataset: {name} from Kaggle...")
            try:
                # Kaggle datasets are usually downloaded as zip files containing CSVs or JSONs.
                # We use the kaggle CLI tool via subprocess.
                dataset_path = os.path.join(self.download_dir, name.split('/')[-1])
                
                # Check if already downloaded to save time
                if not os.path.exists(dataset_path):
                     print(f"Downloading {name} to {dataset_path}...")
                     subprocess.run(
                         ["kaggle", "datasets", "download", "-d", name, "-p", dataset_path, "--unzip"], 
                         check=True
                     )
                     print(f"Successfully downloaded {name}")
                else:
                     print(f"Dataset {name} already exists locally.")

                # After downloading, you would typically read the CSV/JSON files using pandas.
                # Example (you'll need to adjust filename based on actual Kaggle dataset contents):
                # df = pd.read_csv(os.path.join(dataset_path, "ner_dataset.csv"), encoding="latin1")
                # self.datasets[name] = df
                
                self.datasets[name] = f"Local path: {dataset_path}" # Placeholder

            except subprocess.CalledProcessError as e:
                print(f"Error downloading Kaggle dataset {name}: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

    def preprocess(self):
        """Preprocesses the datasets for token classification."""
        for name, data in self.datasets.items():
            print(f"Preprocessing dataset: {name}")
            # TODO: Implement Kaggle dataset specific parsing and Hugging Face Tokenizer logic here.
            # Kaggle NER datasets often come as CSVs with 'Sentence #', 'Word', 'POS', 'Tag'.
            # You'll need to group them into sentences and align tokens.
            pass
        return self.datasets
