Python 3.9.0 (tags/v3.9.0:9cf6752, Oct  5 2020, 15:34:40) [MSC v.1927 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>># Corresponding Section: 5.2 Algorithm 2 Incremental Decision Re-evaluation Algorithm, Definition 11-12
import pandas as pd
import numpy as np
from pydmn import DmnModel, evaluate_decision
from typing import Dict, List, Tuple, Optional
from src.data_dependency_graph import ProcessDataDependencyGraph
from src.ccpa_algorithm import context_aware_change_propagation
from configs.experiment_config import GLOBAL_CONFIG

class DMNDecisionReEvaluator:
    """
    DMN Decision Incremental Re-evaluation class.
    Detects valid environmental changes, performs incremental decision re-evaluation, and handles message boundary events.
    """
    def __init__(self, dmn_model_path: str = None, dmn_model: DmnModel = None):
        if dmn_model_path is not None:
            self.dmn_model = DmnModel.from_file(dmn_model_path)
        elif dmn_model is not None:
            self.dmn_model = dmn_model
        else:
            raise ValueError("Either dmn_model_path or dmn_model must be provided")
        
        # Extract decision metadata from the DMN model
        self.decision_ids: List[str] = [dec.id for dec in self.dmn_model.decisions]
        self.decision_inputs: Dict[str, List[str]] = {dec.id: dec.inputs for dec in self.dmn_model.decisions}
        self.decision_outputs: Dict[str, str] = {dec.id: dec.output for dec in self.dmn_model.decisions}

    def detect_valid_environment_change(
        self,
        env_series: pd.Series,
        normal_threshold: float,
        time_window: Tuple[int, int]
    ) -> bool:
        """
        Valid environmental change detection.
        A change is valid only if the abnormal state persists for K consecutive events.

        Parameters
        ----------
        env_series : pd.Series
            Discretized environmental state series ('normal'/'abnormal')
        normal_threshold : float
            Threshold for normal operation range
        time_window : Tuple[int, int]
            Time window for detection (start index, end index)

        Returns
        -------
        bool
            True if a valid environmental change is detected, False otherwise
        """
        window_data = env_series.iloc[time_window[0]:time_window[1]]
        abnormal_count = (window_data == "abnormal").sum()
        return abnormal_count >= GLOBAL_CONFIG["abnormal_coverage_k"]

    def sac_consistency_check(self, decision_id: str, actual_inputs: Dict) -> bool:
        """
        Service Adherence Criterion (SAC) consistency check.
        Verifies that the actual inputs satisfy the decision's required input interface.

        Parameters
        ----------
        decision_id : str
            ID of the decision to check
        actual_inputs : Dict
            Actual input data provided by the process

        Returns
        -------
        bool
            True if SAC is satisfied, False otherwise
        """
        required_inputs = set(self.decision_inputs[decision_id])
        actual_input_keys = set(actual_inputs.keys())
        return required_inputs.issubset(actual_input_keys)

    def incremental_decision_reevaluation(
        self,
        trace_df: pd.DataFrame,
        changed_env_vars: Dict[str, pd.Series],
        ddg: ProcessDataDependencyGraph,
        injection_event_idx: int,
        original_decision_outputs: Dict[str, Dict[int, object]]
    ) -> Tuple[Dict[str, Dict], List[Dict]]:
        """
        Incremental Decision Re-evaluation Algorithm.
        Performs incremental re-evaluation only for decisions affected by valid environmental changes.

        Parameters
        ----------
        trace_df : pd.DataFrame
            DataFrame of a single process trace
        changed_env_vars : Dict[str, pd.Series]
            Dictionary of changed environmental variables {var_name: state_series}
        ddg : ProcessDataDependencyGraph
            Pre-constructed DDG instance
        injection_event_idx : int
            Index of the event where the environmental change was injected
        original_decision_outputs : Dict[str, Dict[int, object]]
            Original decision outputs cached from baseline execution

        Returns
        -------
        Tuple[Dict[str, Dict], List[Dict]]
            1. updated_outputs: Dictionary of decision outputs that changed after re-evaluation
            2. triggered_boundary_events: List of boundary events triggered by the change
        """
        # Step 1: Initialization
        updated_outputs: Dict[str, Dict] = {}
        triggered_boundary_events: List[Dict] = []
        env_input_set: Dict[str, str] = {}
        env_data_nodes: Set[str] = set()

        # Step 2: Valid Environmental Change Detection
        valid_change_detected = False
        for env_var, env_series in changed_env_vars.items():
            normal_threshold = np.quantile(env_series, GLOBAL_CONFIG["continuous_threshold_quantile"])
            if self.detect_valid_environment_change(
                env_series,
                normal_threshold,
                (injection_event_idx, injection_event_idx + GLOBAL_CONFIG["abnormal_coverage_k"])
            ):
                valid_change_detected = True
                from utils.preprocessing_utils import semantic_discretization
                discretized_state = semantic_discretization(
                    env_series, normal_threshold, is_continuous=True
                ).iloc[injection_event_idx]
                env_input_set[env_var] = discretized_state
                if env_var in ddg.G.nodes:
                    env_data_nodes.add(env_var)
        
        if not valid_change_detected:
            return updated_outputs, triggered_boundary_events

        # Step 3: Map Environmental Changes to Impacted Decisions
        affected_set, _ = context_aware_change_propagation(ddg, trace_df, list(env_data_nodes), injection_event_idx)
        affected_decision_activities = [
            node for node in affected_set
            if ddg.node_types.get(node) == "activity" and node in self.decision_ids
        ]

        # Step 4: Incremental Decision Re-evaluation
        for decision_id in affected_decision_activities:
            decision_event_idx = trace_df[trace_df["activity"] == decision_id].index.min()
            if pd.isna(decision_event_idx) or decision_event_idx < injection_event_idx:
                continue

            original_inputs = trace_df.iloc[decision_event_idx].to_dict()
            updated_inputs = {**original_inputs, **env_input_set}

            if not self.sac_consistency_check(decision_id, updated_inputs):
                triggered_boundary_events.append({
                    "decision_id": decision_id,
                    "event_type": "interface_mismatch",
                    "is_interrupting": False,
                    "message": "SAC consistency check failed: required inputs missing"
                })
                continue

            try:
                new_output = evaluate_decision(self.dmn_model, decision_id, updated_inputs)
            except Exception as e:
                triggered_boundary_events.append({
                    "decision_id": decision_id,
                    "event_type": "rule_coverage_missing",
                    "is_interrupting": False,
                    "error": str(e)
                })
                continue

            original_output = original_decision_outputs.get(decision_id, {}).get(decision_event_idx, None)
            if original_output is not None and new_output != original_output:
                updated_outputs[decision_id] = {
                    "event_idx": decision_event_idx,
                    "original_output": original_output,
                    "updated_output": new_output,
                    "inputs": updated_inputs
                }
                triggered_boundary_events.append({
                    "decision_id": decision_id,
                    "event_type": "output_updated",
                    "is_interrupting": False,
                    "output_change": (original_output, new_output)
                })

        return updated_outputs, triggered_boundary_events
