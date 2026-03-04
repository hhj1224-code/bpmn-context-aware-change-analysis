Python 3.9.0 (tags/v3.9.0:9cf6752, Oct  5 2020, 15:34:40) [MSC v.1927 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>>"""
run_experiment.py
Main experiment script for one-click reproduction of all results in the paper
Corresponds to full experimental pipeline in Section 6 of the paper
"""
import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Import custom modules
from configs.config import EXPERIMENT_CONFIG, DATASET_CONFIG, SEED
from data.download_datasets import download_all_datasets
from data.dataset_pipeline import generate_context_enhanced_logs, inject_changes, generate_ground_truth
from utils.preprocessing_utils import load_event_log, preprocess_environmental_data
from utils.metrics import calculate_classification_metrics
from src.data_dependency_graph import build_data_dependency_graph
from src.ccpa_algorithm import CCPA
from src.incremental_reevaluation import incremental_decision_reevaluation
from src.consistency_analysis import SAC_consistency_check, classify_impact_type
from src.baselines import DIS_Baseline, DTM_Baseline, LSTM_PPM_Baseline, BINet_Baseline

# Fix random seed for full reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Create output directories
os.makedirs("results/figures", exist_ok=True)
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce experiments for context-aware change impact analysis")
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "RTF", "Sepsis", "BPIC2012", "Synthetic"], help="Dataset to run")
    parser.add_argument("--scenario", type=str, default="all", choices=["all", "S1", "S2", "S3"], help="Change scenario to run")
    parser.add_argument("--injection_ratio", type=float, default=None, help="Injection ratio (0.05, 0.1, 0.2)")
    parser.add_argument("--ablation", type=bool, default=False, help="Run robustness ablation study")
    return parser.parse_args()

def run_single_experiment(dataset_name, scenario, injection_ratio, config):
    """
    Run single experiment for a given dataset, scenario and injection ratio
    """
    print(f"\n=== Running Experiment: Dataset={dataset_name}, Scenario={scenario}, Injection Ratio={injection_ratio} ===")
    
    # Step 1: Load and preprocess data
    if dataset_name == "Synthetic":
        log, process_model = load_event_log(dataset_name, synthetic=True, config=config)
    else:
        log, process_model = load_event_log(dataset_name, synthetic=False, config=config)
    
    # Step 2: Context enhancement and environmental data preprocessing
    enhanced_log = preprocess_environmental_data(log, dataset_name, config)
    enhanced_log = generate_context_enhanced_logs(enhanced_log, dataset_name, config)
    
    # Step 3: Inject controlled changes and generate ground truth
    injected_log, injection_info = inject_changes(enhanced_log, scenario, injection_ratio, config)
    ground_truth = generate_ground_truth(injected_log, injection_info, dataset_name, config)
    
    # Step 4: Build Data Dependency Graph (DDG)
    ddg = build_data_dependency_graph(process_model, injected_log, config)
    
    # Step 5: Run our proposed method
    print("Running proposed method...")
    y_true = []
    y_pred_ours = []
    impact_types = []
    
    for trace in tqdm(injected_log, desc="Processing traces"):
        # Ground truth label
        y_true.append(ground_truth[trace.attributes["concept:name"]]["label"])
        
        # CCPA algorithm for change propagation
        affected_set = CCPA(trace, ddg, config)
        
        # Incremental decision re-evaluation for environmental changes
        if scenario in ["S2", "S3"]:
            updated_decisions = incremental_decision_reevaluation(trace, ddg, config)
            affected_set.update([d["decision_activity"] for d in updated_decisions])
        
        # Consistency check and impact type classification
        sac_result = SAC_consistency_check(trace, affected_set, config)
        impact_type = classify_impact_type(sac_result)
        impact_types.append(impact_type)
        
        # Prediction label
        y_pred_ours.append(1 if len(affected_set) > 0 else 0)
    
    # Step 6: Run baseline methods
    print("Running baseline methods...")
    baselines = {
        "DIS": DIS_Baseline(config),
        "DTM": DTM_Baseline(config),
        "LSTM-PPM": LSTM_PPM_Baseline(config),
        "BINet": BINet_Baseline(config)
    }
    
    baseline_results = {}
    for name, baseline in baselines.items():
        print(f"Running {name} baseline...")
        baseline.fit(injected_log, ground_truth)
        y_pred_baseline = baseline.predict(injected_log)
        baseline_results[name] = calculate_classification_metrics(y_true, y_pred_baseline)
    
    # Step 7: Calculate metrics for our method
    ours_metrics = calculate_classification_metrics(y_true, y_pred_ours)
    all_results = {"Ours": ours_metrics, **baseline_results}
    
    # Step 8: Save results
    result_df = pd.DataFrame(all_results).T
    result_df.to_csv(f"results/{dataset_name}_{scenario}_{injection_ratio}_results.csv")
    
    return all_results, impact_types

def main():
    args = parse_args()
    config = EXPERIMENT_CONFIG
    
    # Step 1: Download all datasets if not exists
    print("Step 1: Downloading datasets...")
    download_all_datasets()
    
    # Step 2: Configure experiment parameters
    datasets = ["RTF", "Sepsis", "BPIC2012", "Synthetic"] if args.dataset == "all" else [args.dataset]
    scenarios = ["S1", "S2", "S3"] if args.scenario == "all" else [args.scenario]
    injection_ratios = [0.05, 0.1, 0.2] if args.injection_ratio is None else [args.injection_ratio]
    
    # Step 3: Run all experiments
    full_f1_results = []
    full_s2_results = []
    all_impact_types = []
    
    for dataset in datasets:
        for scenario in scenarios:
            for ratio in injection_ratios:
                results, impact_types = run_single_experiment(dataset, scenario, ratio, config)
                
                # Collect F1-score results
                for method, metrics in results.items():
                    full_f1_results.append({
                        "dataset": dataset,
                        "scenario": scenario,
                        "injection_ratio": ratio,
                        "method": method,
                        "f1": metrics["f1"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "auc": metrics["auc"]
                    })
                
                # Collect S2 scenario results for Table 9
                if scenario == "S2" and ratio == 0.1:
                    for method, metrics in results.items():
                        full_s2_results.append({
                            "dataset": dataset,
                            "method": method,
                            "precision": metrics["precision"],
                            "recall": metrics["recall"],
                            "auc": metrics["auc"]
                        })
                
                # Collect impact types for Synthetic dataset
                if dataset == "Synthetic":
                    for i, impact_type in enumerate(impact_types):
                        all_impact_types.append({
                            "scenario": scenario,
                            "injection_ratio": ratio,
                            "trace_id": i,
                            "impact_type": impact_type
                        })
    
    # Step 4: Save full results
    pd.DataFrame(full_f1_results).to_csv("results/f1_score_comparison.csv", index=False)
    pd.DataFrame(full_s2_results).to_csv("results/precision_recall_auc_S2.csv", index=False)
    pd.DataFrame(all_impact_types).to_csv("results/impact_type_distribution.csv", index=False)
    
    # Step 5: Generate plots
    print("Generating plots...")
    # F1-score comparison plot
    f1_df = pd.DataFrame(full_f1_results)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=f1_df, x="dataset", y="f1", hue="method", ci=None)
    plt.title("F1-Score Comparison Across Datasets")
    plt.ylim(0, 1)
    plt.savefig("results/figures/f1_score_comparison.png", dpi=300, bbox_inches="tight")
    
    # Impact type distribution plot
    if len(all_impact_types) > 0:
        impact_df = pd.DataFrame(all_impact_types)
        impact_dist = impact_df.groupby(["scenario", "injection_ratio", "impact_type"]).size().unstack(fill_value=0)
        impact_dist = impact_dist.div(impact_dist.sum(axis=1), axis=0) * 100
        impact_dist.plot(kind="bar", stacked=True, figsize=(12, 6))
        plt.title("Impact Type Distribution Across Scenarios")
        plt.ylabel("Percentage (%)")
        plt.savefig("results/figures/impact_type_distribution.png", dpi=300, bbox_inches="tight")
    
    print("\n=== All experiments completed successfully! Results saved to 'results/' folder ===")

if __name__ == "__main__":
    main()
