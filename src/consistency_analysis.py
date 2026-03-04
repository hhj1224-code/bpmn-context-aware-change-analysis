Python 3.9.0 (tags/v3.9.0:9cf6752, Oct  5 2020, 15:34:40) [MSC v.1927 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> """
src/consistency_analysis.py
SAC consistency check and impact type classification
Corresponds to paper Section 5.3, implements SAC criterion and 4 impact types
"""
from pm4py.objects.log.obj import Trace
from models.dmn_executor import load_manufacturing_decision_model
from configs.config import EXPERIMENT_CONFIG

def SAC_consistency_check(trace: Trace, affected_set, config=EXPERIMENT_CONFIG):
    """
    Service Adherence Criterion (SAC) consistency check, paper Definition 13
    Args:
        trace: PM4Py Trace object
        affected_set: Set of affected elements from CCPA
        config: Global experiment config
    Returns:
        sac_result: Dict with SAC check results
    """
    # Load decision models
    raw_material_inspection, production_mode_decision = load_manufacturing_decision_model()
    decision_models = [raw_material_inspection, production_mode_decision]
    
    sac_satisfied = True
    decision_output_changed = False
    rule_coverage_issue = False
    interface_mismatch = False
    cascade_propagation = False
    
    # Check each decision invocation in the trace
    for event in trace:
        activity_name = event["concept:name"]
        is_decision_activity = "inspection" in activity_name.lower() or "decision" in activity_name.lower() or "manufacturing" in activity_name.lower()
        
        if not is_decision_activity:
            continue
        
        # Get corresponding decision model
        if "inspection" in activity_name.lower():
            decision_model = raw_material_inspection
        else:
            decision_model = production_mode_decision
        
        # Get actual input data provided by the process
        actual_input = {}
        for input_col in decision_model.input_columns:
            if input_col in event:
                actual_input[input_col] = event[input_col]
        
        # SAC check: actual input must include all required inputs (paper Definition 13)
        required_inputs = decision_model.required_inputs
        missing_inputs = required_inputs - set(actual_input.keys())
        
        if len(missing_inputs) > 0:
            sac_satisfied = False
            interface_mismatch = True
        
        # Evaluate decision to check rule coverage
        output, is_covered = decision_model.evaluate(actual_input)
        if not is_covered:
            rule_coverage_issue = True
        
        # Check if decision output changed
        original_output = event.get("decision_output", None)
        if original_output is not None and output is not None:
            if output != original_output:
                decision_output_changed = True
        
        # Check cascade propagation: output is input to another decision
        if output is not None:
            for other_decision in decision_models:
                if other_decision.decision_id == decision_model.decision_id:
                    continue
                if any(col in decision_model.output_columns for col in other_decision.input_columns):
                    cascade_propagation = True
    
    return {
        "sac_satisfied": sac_satisfied,
        "decision_output_changed": decision_output_changed,
        "rule_coverage_issue": rule_coverage_issue,
        "interface_mismatch": interface_mismatch,
        "cascade_propagation": cascade_propagation
    }

def classify_impact_type(sac_result):
    """
    Classify change impact into 4 types, matches paper Table 4
    Args:
        sac_result: Result from SAC_consistency_check
    Returns:
        impact_type: Integer 1-4, corresponding to the 4 impact types
    """
    sac_satisfied = sac_result["sac_satisfied"]
    rule_coverage_issue = sac_result["rule_coverage_issue"]
    interface_mismatch = sac_result["interface_mismatch"]
    cascade_propagation = sac_result["cascade_propagation"]
    decision_output_changed = sac_result["decision_output_changed"]
    
    # Impact Type 3: Decision interface mismatch (highest priority)
    if interface_mismatch:
        return 3
    
    # Impact Type 2: Decision rule coverage insufficient
    if sac_satisfied and rule_coverage_issue:
        return 2
    
    # Impact Type 4: Cross-decision cascade propagation
    if sac_satisfied and cascade_propagation and decision_output_changed:
        return 4
    
    # Impact Type 1: Only affects process execution
    if sac_satisfied and not rule_coverage_issue:
        return 1
    
    # Default to Type 1 if no other type matches
    return 1