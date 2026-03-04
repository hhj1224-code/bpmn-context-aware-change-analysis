Python 3.9.0 (tags/v3.9.0:9cf6752, Oct  5 2020, 15:34:40) [MSC v.1927 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>># Corresponding Section: 6.1 Context-Enhanced Log, 6.2 Controlled Change Injection
import pandas as pd
import numpy as np
from typing import Tuple, Dict
from configs.experiment_config import GLOBAL_CONFIG, DATASET_ENV_CONFIG, DATASET_INJECT_ATTRS
from utils.preprocessing_utils import sliding_window_preprocess, semantic_discretization

np.random.seed(GLOBAL_CONFIG["random_seed"])

def enhance_log_with_environmental_context(
    log_df: pd.DataFrame,
    dataset_name: str
) -> Tuple[pd.DataFrame, Dict]:
    """Enhances the event log with environmental variables and context states (Table 6)."""
    enhanced_df = log_df.copy()
    env_metadata = DATASET_ENV_CONFIG[dataset_name]

    for env_var in env_metadata:
        var_name = env_var["name"]
        if env_var["type"] == "continuous":
            enhanced_df[var_name] = np.random.normal(loc=25, scale=5, size=len(enhanced_df))
        else:
            categories = list(env_var["mapping"].keys())
            probabilities = [0.9, 0.1] if len(categories) == 2 else [0.25]*len(categories)
            enhanced_df[var_name] = np.random.choice(categories, size=len(enhanced_df), p=probabilities)

    for env_var in env_metadata:
        var_name = env_var["name"]
        logged_field = env_var["logged_field"]
        if env_var["type"] == "continuous":
            smoothed = sliding_window_preprocess(
                enhanced_df[var_name],
                enhanced_df["timestamp"],
                window_minutes=GLOBAL_CONFIG["window_length_minutes"]
            )
            normal_threshold = np.quantile(smoothed, GLOBAL_CONFIG["continuous_threshold_quantile"])
            enhanced_df[logged_field] = semantic_discretization(
                smoothed, normal_threshold, is_continuous=True
            )
        else:
            enhanced_df[logged_field] = semantic_discretization(
                enhanced_df[var_name],
                None,
                is_continuous=False,
                discrete_mapping=env_var["mapping"]
            )

    return enhanced_df, env_metadata

def inject_controlled_changes(
    log_df: pd.DataFrame,
    dataset_name: str,
    scenario: str,
    injection_ratio: float,
    env_metadata: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Injects controlled changes into the event log for 3 scenarios (S1/S2/S3)."""
    injected_df = log_df.copy()
    case_ids = injected_df["case_id"].unique()
    num_inject_cases = int(len(case_ids) * injection_ratio)
    inject_case_ids = np.random.choice(case_ids, size=num_inject_cases, replace=False)

    ground_truth = pd.DataFrame({
        "case_id": case_ids,
        "Affected": 0,
        "impact_type": 0
    }).set_index("case_id")

    business_attrs = DATASET_INJECT_ATTRS[dataset_name]
    business_attrs = [attr for attr in business_attrs if attr in injected_df.columns]
    env_attrs = [meta["logged_field"] for meta in env_metadata]

    for case_id in inject_case_ids:
        case_mask = injected_df["case_id"] == case_id
        case_events = injected_df[case_mask].reset_index()
        trace_length = len(case_events)
        if trace_length < 5:
            continue

        min_inject_idx = int(trace_length * GLOBAL_CONFIG["injection_point_range"][0])
        max_inject_idx = int(trace_length * GLOBAL_CONFIG["injection_point_range"][1])
        injection_point = np.random.randint(min_inject_idx, max_inject_idx)
        injection_event_id = case_events.iloc[injection_point]["index"]

        is_affected = False
        impact_type = 0

        if scenario == "S1" and len(business_attrs) > 0:
            inject_attr = np.random.choice(business_attrs)
            original_value = injected_df.loc[injection_event_id, inject_attr]
            value_quantiles = np.quantile(injected_df[inject_attr].dropna(), [0.25, 0.5, 0.75])
            if original_value <= value_quantiles[0]:
                new_value = value_quantiles[2]
            elif original_value <= value_quantiles[1]:
                new_value = value_quantiles[2] * 1.5
            else:
                new_value = value_quantiles[0] * 0.5
            inject_mask = (
                (injected_df["case_id"] == case_id) &
                (injected_df.index >= injection_event_id) &
                (injected_df.index <= injection_event_id + GLOBAL_CONFIG["abnormal_coverage_k"])
            )
            injected_df.loc[inject_mask, inject_attr] = new_value
            is_affected = True
            impact_type = 1

        elif scenario == "S2" and len(env_attrs) > 0:
            inject_attr = np.random.choice(env_attrs)
            inject_mask = (
                (injected_df["case_id"] == case_id) &
                (injected_df.index >= injection_event_id) &
                (injected_df.index <= injection_event_id + GLOBAL_CONFIG["abnormal_coverage_k"])
            )
            injected_df.loc[inject_mask, inject_attr] = "abnormal"
            is_affected = True
            impact_type = 3

        elif scenario == "S3" and len(business_attrs) > 0 and len(env_attrs) > 0:
            business_inject_attr = np.random.choice(business_attrs)
            original_value = injected_df.loc[injection_event_id, business_inject_attr]
            value_quantiles = np.quantile(injected_df[business_inject_attr].dropna(), [0.25, 0.5, 0.75])
            new_value = value_quantiles[2] * 1.5 if original_value <= value_quantiles[1] else value_quantiles[0] * 0.5
            inject_mask = (
                (injected_df["case_id"] == case_id) &
                (injected_df.index >= injection_event_id) &
                (injected_df.index <= injection_event_id + GLOBAL_CONFIG["abnormal_coverage_k"])
            )
            injected_df.loc[inject_mask, business_inject_attr] = new_value

            env_inject_attr = np.random.choice(env_attrs)
            injected_df.loc[inject_mask, env_inject_attr] = "abnormal"

            is_affected = True
            impact_type = 4

        if is_affected:
            ground_truth.loc[case_id, "Affected"] = 1
            ground_truth.loc[case_id, "impact_type"] = impact_type

    return injected_df, ground_truth.reset_index()
