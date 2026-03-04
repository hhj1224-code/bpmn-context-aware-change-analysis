Python 3.9.0 (tags/v3.9.0:9cf6752, Oct  5 2020, 15:34:40) [MSC v.1927 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> """
models/dmn_executor.py
DMN decision execution engine, implements DMN decision table evaluation
Corresponds to paper Fig.3 manufacturing decision model
"""
import pandas as pd
import numpy as np

class DMNDecisionTable:
    """
    DMN Decision Table implementation, matches OMG DMN standard
    """
    def __init__(self, decision_id, decision_name, input_columns, output_columns, rules):
        """
        Initialize DMN Decision Table
        Args:
            decision_id: Unique ID of the decision
            decision_name: Name of the decision
            input_columns: List of input column names (decision inputs)
            output_columns: List of output column names (decision outputs)
            rules: List of decision rules, each rule is a dict with input conditions and output values
        """
        self.decision_id = decision_id
        self.decision_name = decision_name
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.rules = rules
        self.required_inputs = set(input_columns)
    
    def evaluate_rule(self, rule, input_data):
        """
        Evaluate a single decision rule against input data
        Args:
            rule: Decision rule dict
            input_data: Dict of input data
        Returns:
            is_matched: Boolean indicating if the rule matches the input data
            output: Output values if matched, None otherwise
        """
        # Check all input conditions
        for input_col in self.input_columns:
            input_value = input_data.get(input_col, None)
            rule_condition = rule["input"].get(input_col, None)
            
            # Skip if no condition for this input
            if rule_condition is None:
                continue
            
            # Handle None input value
            if input_value is None:
                return False, None
            
            # Evaluate condition
            if isinstance(rule_condition, str) and rule_condition.startswith(">"):
                threshold = float(rule_condition[1:])
                if not (input_value > threshold):
                    return False, None
            elif isinstance(rule_condition, str) and rule_condition.startswith("<"):
                threshold = float(rule_condition[1:])
                if not (input_value < threshold):
                    return False, None
            elif isinstance(rule_condition, str) and "(" in rule_condition and "]" in rule_condition:
                # Range condition, e.g., "(30,45]"
                range_str = rule_condition.replace("(", "").replace("]", "")
                lower, upper = map(float, range_str.split(","))
                if not (lower < input_value <= upper):
                    return False, None
            else:
                # Exact match
                if input_value != rule_condition:
                    return False, None
        
        # All conditions matched, return output
        return True, rule["output"]
    
    def evaluate(self, input_data):
        """
        Evaluate the decision table against input data (hit policy: FIRST)
        Args:
            input_data: Dict of input data
        Returns:
            output: Decision output dict if matched, None otherwise
            is_covered: Boolean indicating if the input is covered by any rule
        """
        for rule in self.rules:
            is_matched, output = self.evaluate_rule(rule, input_data)
            if is_matched:
                return output, True
        
        # No rule matched
        return None, False

# Predefined manufacturing decision model (matches paper Fig.3)
def load_manufacturing_decision_model():
    """
    Load the manufacturing decision model from paper Fig.3
    Returns:
        raw_material_inspection: DMNDecisionTable for raw material inspection
        production_mode_decision: DMNDecisionTable for production mode decision
    """
    # Raw Material Inspection Decision Table (paper Fig.3 left)
    inspection_rules = [
        {
            "input": {"raw_material_quantity": "Sufficient", "raw_material_specification": True},
            "output": {"inspection_result": "Accept", "raw_material_supply": True}
        },
        {
            "input": {"raw_material_quantity": "Sufficient", "raw_material_specification": False},
            "output": {"inspection_result": "Reject", "raw_material_supply": False}
        },
        {
            "input": {"raw_material_quantity": "Shortage", "raw_material_specification": True},
            "output": {"inspection_result": "Accept", "raw_material_supply": False}
        }
    ]
    
    raw_material_inspection = DMNDecisionTable(
        decision_id="raw_material_inspection",
        decision_name="Raw Material Inspection",
        input_columns=["raw_material_quantity", "raw_material_specification"],
        output_columns=["inspection_result", "raw_material_supply"],
        rules=inspection_rules
    )
    
    # Production Mode Decision Table (paper Fig.3 right)
    production_rules = [
        {
            "input": {"order_quantity": "(30,45]", "raw_material_supply": True, "machine_status": "Well", "delivery_date": "<=4"},
            "output": {"production_mode": "Single Line"}
        },
        {
            "input": {"order_quantity": "<=30", "raw_material_supply": False, "delivery_date": "<=30"},
            "output": {"production_mode": "Outsourced"}
        },
        {
            "input": {"order_quantity": ">45", "raw_material_supply": True, "machine_status": "Well"},
            "output": {"production_mode": "Single Line"}
        },
        {
            "input": {"order_quantity": "(4,6]", "raw_material_supply": True, "machine_status": "Faults", "delivery_date": "(30,45]"},
            "output": {"production_mode": "Multiple Lines"}
        },
        {
            "input": {"order_quantity": "<=30", "raw_material_supply": False, "delivery_date": "<=30"},
            "output": {"production_mode": "Outsourced"}
        },
        {
            "input": {"order_quantity": ">6", "raw_material_supply": True, "machine_status": "Well", "delivery_date": "(30,45]"},
            "output": {"production_mode": "Multiple Lines"}
        },
        {
            "input": {"raw_material_supply": False, "delivery_date": ">45"},
            "output": {"production_mode": "Outsourced"}
        }
    ]
    
    production_mode_decision = DMNDecisionTable(
        decision_id="production_mode_decision",
        decision_name="Production Mode Decision",
        input_columns=["order_quantity", "raw_material_supply", "machine_status", "delivery_date"],
        output_columns=["production_mode"],
        rules=production_rules
    )
    
    return raw_material_inspection, production_mode_decision