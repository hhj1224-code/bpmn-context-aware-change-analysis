Python 3.9.0 (tags/v3.9.0:9cf6752, Oct  5 2020, 15:34:40) [MSC v.1927 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> """
src/incremental_reevaluation.py
Incremental decision re-evaluation algorithm
Corresponds to paper Section 5.2 Algorithm 2
"""
import numpy as np
from pm4py.objects.log.obj import Trace
from src.data_dependency_graph import DataDependencyGraph
from src.ccpa_algorithm import CCPA
from models.dmn_executor import load_manufacturing_decision_model
from configs.config import EXPERIMENT_CONFIG

def detect_effective_environmental_change(trace: Trace, env_var_config, config=EXPERIMENT_CONFIG):
    """
    Detect effective environmental change, paper Definition 11
    Args:
        trace: PM4Py Trace object
        env_var_config: Environmental variable configuration
        config: Global experiment config
    Returns:
        has_effective_change: Boolean indicating if effective change occurred
        changed_env_vars: List of changed environmental variables
        change_timestamps: Timestamps of the changes
    """
    logged_field = env_var_config["logged_field"]
    normal_state = config["default_env_state_normal"]
    abnormal_state = config["default_env_state_abnormal"]
    
    # Extract environmental states from trace
    env_states = []
    timestamps = []
    for event in trace:
        if logged_field in event:
            env_states.append(event[logged_field])
            timestamps.append(event["time:timestamp"])
    
    if len(env_states) == 0:
        return False, [], []
    
    # Detect state transition from normal to abnormal
    has_effective_change = False
    changed_env_vars = []
    change_timestamps = []
    
    previous_state = env_states[0]
    for i, current_state in enumerate(env_states):
        if previous_state == normal_state and current_state == abnormal_state:
            has_effective_change = True
            changed_env_vars.append(env_var_config["name"])
            change_timestamps.append(timestamps[i])
        previous_state = current_state
    
    return has_effective_change, changed_env_vars, change_timestamps

def incremental_decision_reevaluation(trace: Trace, ddg: DataDependencyGraph, config=EXPERIMENT_CONFIG):
    """
    Incremental decision re-evaluation algorithm, paper Section 5.2 Algorithm 2
    Args:
        trace: PM4Py Trace object
        ddg: DataDependencyGraph object
        config: Global experiment config
    Returns:
        updated_decisions: List of updated decision outputs
    """
    # Step 1: Initialize variables (Algorithm 2 line 1)
    env_input_set = set()
    env_data_nodes = set()
    updated_decision_outputs = []
    
    # Step 2: Load DMN decision models
    raw_material_inspection, production_mode_decision = load_manufacturing_decision_model()
    decision_models = {
        "raw_material_inspection": raw_material_inspection,
        "production_mode_decision": production_mode_decision
    }
    
    # Step 3: Detect effective environmental changes (Algorithm 2 lines 2-5)
    dataset_name = trace.attributes["dataset"] if "dataset" in trace.attributes else "Synthetic"
    from configs.config import DATASET_CONFIG
    env_variables = DATASET_CONFIG[dataset_name]["environmental_variables"]
    
    has_any_effective_change = False
    all_changed_env_vars = []
    
    for env_var in env_variables:
        has_change, changed_vars, _ = detect_effective_environmental_change(trace, env_var, config)
        if has_change:
            has_any_effective_change = True
            all_changed_env_vars.extend(changed_vars)
    
    # No effective change, return empty list
    if not has_any_effective_change:
        return updated_decision_outputs
    
    # Step 4: Process changed environmental variables (Algorithm 2 lines 6-11)
    for env_var in env_variables:
        if env_var["name"] not in all_changed_env_vars:
            continue
        
        logged_field = env_var["logged_field"]
        # Discretize environmental data (already done in preprocessing)
        discrete_state = config["default_env_state_abnormal"]
        env_input_set.add((env_var["name"], discrete_state))
        
        # Map environmental variable to data nodes in DDG
        if logged_field in ddg.data_nodes:
            env_data_nodes.add(logged_field)
    
    # Step 5: Run CCPA to get affected set (Algorithm 2 lines 12-13)
    affected_set = CCPA(trace, ddg, config)
    affected_set.update(env_data_nodes)
    
    # Identify affected decision activities
    decision_activities = [
        node for node in affected_set 
        if ddg.get_node_type(node) == "activity" and "inspection" in node.lower() or "decision" in node.lower() or "manufacturing" in node.lower()
    ]
    
    # Step 6: Incremental re-evaluation of affected decisions (Algorithm 2 lines 14-34)
    # Get executed activities and their completion status
    executed_activities = [event["concept:name"] for event in trace]
    completed_activities = set(executed_activities)
    
    for decision_activity in decision_activities:
        # Skip completed activities (Algorithm 2 lines 15-17)
        if decision_activity in completed_activities:
            continue
        
        # Get decision logic and original input data (Algorithm 2 lines 18-19)
        if "inspection" in decision_activity.lower():
            decision_model = decision_models["raw_material_inspection"]
        else:
            decision_model = decision_models["production_mode_decision"]
        
        # Get original input data from trace
        original_input = {}
        for event in trace:
            if event["concept:name"] == decision_activity:
                for input_col in decision_model.input_columns:
                    if input_col in event:
                        original_input[input_col] = event[input_col]
                break
        
        # Fuse original input with environmental context (Algorithm 2 line 20)
        updated_input = original_input.copy()
        for env_name, env_state in env_input_set:
            if env_name in decision_model.input_columns:
                updated_input[env_name] = env_state
            elif "machine_status" in decision_model.input_columns and env_name == "temp":
                updated_input["machine_status"] = "Faults" if env_state == config["default_env_state_abnormal"] else "Well"
        
        # Execute DMN decision logic (Algorithm 2 lines 21-22)
        new_output, is_covered = decision_model.evaluate(updated_input)
        original_output, _ = decision_model.evaluate(original_input)
        
        # Check if decision output changed (Algorithm 2 lines 23-24)
        if new_output != original_output:
            updated_decision_outputs.append({
                "decision_activity": decision_activity,
                "decision_id": decision_model.decision_id,
                "original_input": original_input,
                "updated_input": updated_input,
                "original_output": original_output,
                "updated_output": new_output,
                "is_covered": is_covered
            })
            
            # Handle boundary events (Algorithm 2 lines 25-33)
            # For interrupting boundary events: terminate running activity
            # For non-interrupting boundary events: trigger parallel path
            # Implemented in BPMN executor
    
    # Step 7: Return updated decision outputs (Algorithm 2 line 35)
    return updated_decision_outputs