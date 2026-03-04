Python 3.9.0 (tags/v3.9.0:9cf6752, Oct  5 2020, 15:34:40) [MSC v.1927 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>># Corresponding Section: 6.3 Experimental Results and Full Experiment Pipeline
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

from configs.experiment_config import GLOBAL_CONFIG, RANDOM_SEED
from data.download_datasets import load_dataset
from data.dataset_pipeline import enhance_log_with_environmental_context, inject_controlled_changes
from src.data_dependency_graph import ProcessDataDependencyGraph
from src.ccpa_algorithm import context_aware_change_propagation
from src.baselines import DISBaseline, DTMBaseline, LSTMPPMBaseline, BINetBaseline
from src.metrics import calculate_metrics

np.random.seed(RANDOM_SEED)
os.makedirs("./results/paper_results/", exist_ok=True)
os.makedirs("./results/figures/", exist_ok=True)

def run_full_experiment():
    """Runs the full experiment pipeline to reproduce all results in the paper."""
    all_results = []
    impact_type_distribution = []

    for dataset_name in GLOBAL_CONFIG["datasets"]:
        print(f"\n{'='*80}")
        print(f"Processing Dataset: {dataset_name}")
        print(f"{'='*80}")

        print(f"[1/5] Loading and preprocessing dataset...")
        log_df = load_dataset(dataset_name)
        enhanced_df, env_metadata = enhance_log_with_environmental_context(log_df, dataset_name)
        
        activities = enhanced_df["activity"].unique()
        data_columns = [
            col for col in enhanced_df.columns
            if col not in ["case_id", "activity", "timestamp", "resource"]
        ]
        activity_io_map = {
            act: {"input": data_columns, "output": data_columns}
            for act in activities
        }
        path_constraint_map = {}
        
        ddg = ProcessDataDependencyGraph(event_log=enhanced_df)
        ddg.build_from_log_and_model(activity_io_map, path_constraint_map)
        print(f"DDG constructed with {len(ddg.G.nodes)} nodes and {len(ddg.G.edges)} edges.")

        print(f"[2/5] Initializing baseline models...")
        dis_model = DISBaseline(activity_io_map, path_constraint_map)
        dtm_model = DTMBaseline()
        lstm_model = LSTMPPMBaseline()
        binet_model = BINetBaseline()
        models = {
            "DIS": dis_model,
            "DTM": dtm_model,
            "LSTM-PPM": lstm_model,
            "BINet": binet_model,
            "Ours": "Proposed Method"
        }

        for scenario in GLOBAL_CONFIG["scenarios"]:
            print(f"\n----- Scenario: {scenario} -----")
            for injection_ratio in GLOBAL_CONFIG["injection_ratios"]:
                print(f"Injection Ratio: {injection_ratio*100}%")
                kf = KFold(n_splits=GLOBAL_CONFIG["k_fold"], shuffle=True, random_state=RANDOM_SEED)
                case_ids = enhanced_df["case_id"].unique()
                fold_metrics = {model_name: {"precision": [], "recall": [], "f1": [], "auc": []} for model_name in models.keys()}

                for fold, (train_idx, test_idx) in enumerate(kf.split(case_ids)):
                    print(f"  Fold {fold+1}/{GLOBAL_CONFIG['k_fold']}")
                    train_case_ids = case_ids[train_idx]
                    test_case_ids = case_ids[test_idx]
                    train_log = enhanced_df[enhanced_df["case_id"].isin(train_case_ids)].copy()
                    test_log = enhanced_df[enhanced_df["case_id"].isin(test_case_ids)].copy()

                    train_injected, train_gt = inject_controlled_changes(
                        train_log, dataset_name, scenario, injection_ratio, env_metadata
                    )
                    test_injected, test_gt = inject_controlled_changes(
                        test_log, dataset_name, scenario, injection_ratio, env_metadata
                    )

                    dtm_model.fit(train_injected, train_gt)
                    lstm_model.fit(train_injected, train_gt)
                    binet_model.fit(train_injected, train_gt)

                    y_true = test_gt.sort_values("case_id")["Affected"].values
                    for model_name, model in models.items():
                        if model_name == "DIS":
                            y_pred = []
                            for case_id in test_gt["case_id"]:
                                case_trace = test_injected[test_injected["case_id"] == case_id]
                                changed_data = []
                                for col in data_columns:
                                    if case_trace[col].nunique() > 1:
                                        changed_data.append(col)
                                pred = model.predict(case_trace, changed_data)
                                y_pred.append(pred)
                            y_pred = np.array(y_pred)

                        elif model_name == "Ours":
                            y_pred = []
                            case_impact_types = []
                            for case_id in test_gt["case_id"]:
                                case_trace = test_injected[test_injected["case_id"] == case_id].reset_index()
                                changed_data = []
                                injection_idx = 0
                                for col in data_columns:
                                    if case_trace[col].nunique() > 1:
                                        changed_data.append(col)
                                        injection_idx = case_trace[case_trace[col].diff() != 0].index.min()
                                affected_set, _ = context_aware_change_propagation(ddg, case_trace, changed_data, injection_idx)
                                pred = 1 if len(affected_set) > 0 else 0
                                y_pred.append(pred)
                                if pred == 1:
                                    impact_type = test_gt[test_gt["case_id"] == case_id]["impact_type"].values[0]
                                    case_impact_types.append(impact_type)
                            y_pred = np.array(y_pred)
                            if scenario == "S3" and injection_ratio == 0.20:
                                impact_type_distribution.extend(case_impact_types)

                        elif model_name == "DTM":
                            y_pred = dtm_model.predict(test_injected)
                        elif model_name == "LSTM-PPM":
                            y_pred = lstm_model.predict(test_injected)
                        elif model_name == "BINet":
                            y_pred = binet_model.predict(test_injected)

                        metrics = calculate_metrics(y_true, y_pred)
                        for metric_name, value in metrics.items():
                            if metric_name in fold_metrics[model_name]:
                                fold_metrics[model_name][metric_name].append(value)

                for model_name in models.keys():
                    avg_precision = np.mean(fold_metrics[model_name]["precision"])
                    avg_recall = np.mean(fold_metrics[model_name]["recall"])
                    avg_f1 = np.mean(fold_metrics[model_name]["f1"])
                    avg_auc = np.mean(fold_metrics[model_name]["auc"])

                    all_results.append({
                        "dataset": dataset_name,
                        "scenario": scenario,
                        "injection_ratio": injection_ratio,
                        "model": model_name,
                        "precision": round(avg_precision, 3),
                        "recall": round(avg_recall, 3),
                        "f1": round(avg_f1, 3),
                        "auc": round(avg_auc, 3)
                    })

                    print(f"    {model_name:10} | F1: {avg_f1:.3f} | Precision: {avg_precision:.3f} | Recall: {avg_recall:.3f} | AUC: {avg_auc:.3f}")

    print(f"\n{'='*80}")
    print("Saving experiment results...")
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("./results/paper_results/full_experiment_results.csv", index=False, encoding="utf-8-sig")
    
    table8 = results_df.pivot_table(
        index=["dataset", "model"],
        columns=["scenario", "injection_ratio"],
        values="f1"
    )
    table8.to_csv("./results/paper_results/table8_f1_score_results.csv", encoding="utf-8-sig")
    print("Table 8 (F1-score) saved to ./results/paper_results/table8_f1_score_results.csv")

    table9 = results_df[(results_df["scenario"] == "S2") & (results_df["injection_ratio"] == 0.10)]
    table9.to_csv("./results/paper_results/table9_s2_metrics.csv", index=False, encoding="utf-8-sig")
    print("Table 9 (S2 scenario metrics) saved to ./results/paper_results/table9_s2_metrics.csv")

    if impact_type_distribution:
        impact_type_df = pd.DataFrame({"impact_type": impact_type_distribution})
        impact_type_df.to_csv("./results/paper_results/impact_type_distribution.csv", index=False, encoding="utf-8-sig")
        print("Impact type distribution (Figure 9) saved to ./results/paper_results/impact_type_distribution.csv")

    print(f"All experiments completed successfully!")
    print(f"Results are saved to ./results/paper_results/")
    print(f"{'='*80}")

    return results_df

if __name__ == "__main__":
    final_results = run_full_experiment()
