Python 3.9.0 (tags/v3.9.0:9cf6752, Oct  5 2020, 15:34:40) [MSC v.1927 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>># Corresponding Section: 6.1 Datasets
import os
import pm4py
import pandas as pd
import numpy as np
from configs.experiment_config import RANDOM_SEED, DATASET_INJECT_ATTRS
from utils.preprocessing_utils import standardize_event_log

np.random.seed(RANDOM_SEED)

DATASET_LINKS = {
    "RTF": "https://data.4tu.nl/ndownloader/files/21514728",
    "Sepsis": "https://data.4tu.nl/ndownloader/files/21514731",
    "BPIC2012": "https://data.4tu.nl/ndownloader/files/21514734"
}

def download_public_datasets(data_dir: str = "./data/") -> None:
    """Automatically download public standard datasets from official DOI links."""
    os.makedirs(data_dir, exist_ok=True)
    
    for dataset_name, url in DATASET_LINKS.items():
        file_path = os.path.join(data_dir, f"{dataset_name}.xes.gz")
        if os.path.exists(file_path):
            print(f"Dataset {dataset_name} already exists, skipping download.")
            continue
        
        print(f"Downloading {dataset_name} dataset from official DOI...")
        try:
            log = pm4py.read_xes(url)
            pm4py.write_xes(log, file_path)
            print(f"Successfully downloaded and saved {dataset_name} dataset.")
        except Exception as e:
            print(f"Failed to download {dataset_name} dataset: {str(e)}")
            print(f"Please manually download the dataset from {url} and save it to {file_path}")

def generate_synthetic_dataset(
    num_cases: int = 10000,
    bpmn_model_path: str = "./models/manufacturing_process.bpmn",
    save_path: str = "./data/Synthetic.csv"
) -> pd.DataFrame:
    """Generate synthetic dataset from the smart manufacturing BPMN model."""
    print(f"Generating synthetic dataset with {num_cases} cases...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if os.path.exists(bpmn_model_path):
        bpmn_model = pm4py.read_bpmn(bpmn_model_path)
    else:
        raise FileNotFoundError(f"BPMN model not found at {bpmn_model_path}")
    
    synthetic_log = pm4py.simulate.play_out(
        bpmn_model,
        num_cases=num_cases,
        random_seed=RANDOM_SEED
    )
    
    synthetic_df = standardize_event_log(pm4py.convert_to_dataframe(synthetic_log))
    
    synthetic_df["order_quantity"] = np.random.randint(1, 10, size=len(synthetic_df))
    synthetic_df["order_type"] = np.random.choice(["A1", "A2", "B1", "B2"], size=len(synthetic_df))
    synthetic_df["delivery_date"] = np.random.randint(15, 60, size=len(synthetic_df))
    synthetic_df["raw_material_quantity"] = np.random.uniform(0.5, 1.2, size=len(synthetic_df)) * synthetic_df["order_quantity"]
    synthetic_df["raw_material_specification"] = np.random.choice([True, False], size=len(synthetic_df), p=[0.9, 0.1])
    
    synthetic_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"Successfully generated synthetic dataset and saved to {save_path}.")
    
    return synthetic_df

def load_dataset(dataset_name: str, data_dir: str = "./data/") -> pd.DataFrame:
    """Load the specified dataset into a standardized DataFrame."""
    if dataset_name == "Synthetic":
        file_path = os.path.join(data_dir, "Synthetic.csv")
        if not os.path.exists(file_path):
            log_df = generate_synthetic_dataset(data_dir=data_dir)
        else:
            log_df = pd.read_csv(file_path)
            log_df = standardize_event_log(log_df)
    else:
        file_path = os.path.join(data_dir, f"{dataset_name}.xes.gz")
        if not os.path.exists(file_path):
            download_public_datasets(data_dir=data_dir)
        log = pm4py.read_xes(file_path)
        log_df = standardize_event_log(pm4py.convert_to_dataframe(log))
    
    return log_df

if __name__ == "__main__":
    download_public_datasets()
    generate_synthetic_dataset()
    print("All datasets are ready!")
