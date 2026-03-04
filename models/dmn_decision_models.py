Python 3.9.0 (tags/v3.9.0:9cf6752, Oct  5 2020, 15:34:40) [MSC v.1927 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> """
Corresponds to Section 4.2 Integrated Modeling & Figure 3: DMN Decision Models for Manufacturing
Implements the two core DMN decision tables (Raw Material Inspection, Manufacturing),
SAC (Service Adherence Criterion) compliance check, and incremental decision re-evaluation
Aligned with Definition 4, 13 in the paper
"""
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, field

# --------------------------
# Core DMN Decision Rule & Model
# --------------------------
@dataclass
class DMNDecisionRule:
    """
    Single DMN decision rule definition
    :param rule_id: Unique identifier of the rule
    :param input_conditions: Dict of {input_attribute: expected_value} for rule matching
    :param output_value: Output value when the rule is matched
    :param description: Human-readable description of the rule
    """
    rule_id: str
    input_conditions: Dict[str, any]
    output_value: any
    description: str = ""

class DMNDecisionModel:
    """
    Base class for DMN Decision Model, aligned with OMG DMN standard and Definition 4 in Section 3.3
    """
    def __init__(
        self,
        decision_id: str,
        decision_name: str,
        required_inputs: Set[str],
        output_attribute: str,
        rules: List[DMNDecisionRule] = None
    ):
        self.decision_id = decision_id
        self.decision_name = decision_name
        self.required_inputs = required_inputs  # For SAC check (Definition 13)
        self.output_attribute = output_attribute
        self.rules = rules if rules is not None else []

    def add_rule(self, rule: DMNDecisionRule):
        """Add a new decision rule to the model"""
        self.rules.append(rule)

    def execute(self, input_data: Dict) -> Tuple[bool, Optional[any]]:
        """
        Execute the DMN decision logic with given input data
        :param input_data: Dict of input data {attr_name: value}
        :return: (execution_success, decision_output)
        """
        # Match the first satisfied rule (hit policy: FIRST)
        for rule in self.rules:
            rule_matched = True
            for attr, expected_val in rule.input_conditions.items():
                if input_data.get(attr) != expected_val:
                    rule_matched = False
                    break
            if rule_matched:
                return True, rule.output_value
        
        # No matching rule found
        return False, None

# --------------------------
# Figure 3: Raw Material Inspection DMN Model
# --------------------------
class RawMaterialInspectionDMN(DMNDecisionModel):
    """
    Raw Material Inspection Decision Model (Left part of Figure 3)
    Aligned with the decision table in the motivation case (Section 3.1)
    """
    def __init__(self):
        super().__init__(
            decision_id="raw_material_inspection",
            decision_name="Raw Material Inspection",
            required_inputs={"raw_material_quantity", "raw_material_specification"},
            output_attribute="inspection_result"
        )
        # Add rules from Figure 3's inspection decision table
        self.add_rule(DMNDecisionRule(
            rule_id="R1",
            input_conditions={"raw_material_quantity": "Sufficient", "raw_material_specification": True},
            output_value="Accept",
            description="Sufficient quantity and valid specification: accept material"
        ))
        self.add_rule(DMNDecisionRule(
            rule_id="R2",
            input_conditions={"raw_material_quantity": "Sufficient", "raw_material_specification": False},
            output_value="Reject",
            description="Sufficient quantity but invalid specification: reject material"
        ))
        self.add_rule(DMNDecisionRule(
            rule_id="R3",
            input_conditions={"raw_material_quantity": "Shortage", "raw_material_specification": True},
            output_value="Accept",
            description="Shortage quantity but valid specification: accept with warning"
        ))
        self.add_rule(DMNDecisionRule(
            rule_id="R4",
            input_conditions={"raw_material_quantity": "Shortage", "raw_material_specification": False},
            output_value="Reject",
            description="Shortage quantity and invalid specification: reject material"
        ))

# --------------------------
# Figure 3: Manufacturing DMN Model
# --------------------------
class ManufacturingDMN(DMNDecisionModel):
    """
    Manufacturing Production Mode Decision Model (Right part of Figure 3)
    Extended with environment context awareness (Section 4.2)
    """
    def __init__(self):
        super().__init__(
            decision_id="manufacturing",
            decision_name="Manufacturing Production Mode",
            required_inputs={"order_quantity", "delivery_date", "inspection_result", "machine_status"},
            output_attribute="production_mode"
        )
        # Add rules from Figure 3's manufacturing decision table, extended with machine status (environment context)
        self.add_rule(DMNDecisionRule(
            rule_id="M1",
            input_conditions={
                "order_quantity": "<=4",
                "delivery_date": "(30,45]",
                "inspection_result": "Accept",
                "machine_status": "Well"
            },
            output_value="Single Line",
            description="Small order, normal delivery, valid material, healthy machine: single line production"
        ))
        self.add_rule(DMNDecisionRule(
            rule_id="M2",
            input_conditions={
                "order_quantity": "<=4",
                "delivery_date": "<=30",
                "inspection_result": "Accept",
                "machine_status": "Well"
            },
            output_value="Outsourced",
            description="Small order, tight delivery: outsourced production"
        ))
        self.add_rule(DMNDecisionRule(
            rule_id="M3",
            input_conditions={
                "order_quantity": "<=4",
                "delivery_date": ">45",
                "inspection_result": "Accept",
                "machine_status": "Well"
            },
            output_value="Single Line",
            description="Small order, loose delivery: single line production"
        ))
        self.add_rule(DMNDecisionRule(
            rule_id="M4",
            input_conditions={
                "order_quantity": "(4,6]",
                "delivery_date": "(30,45]",
                "inspection_result": "Accept",
                "machine_status": "Well"
            },
            output_value="Multiple Lines",
            description="Medium order, normal delivery: multiple lines production"
        ))
        self.add_rule(DMNDecisionRule(
            rule_id="M5",
            input_conditions={
                "order_quantity": "(4,6]",
                "delivery_date": "<=30",
                "inspection_result": "Accept",
                "machine_status": "Well"
            },
            output_value="Outsourced",
            description="Medium order, tight delivery: outsourced production"
        ))
        self.add_rule(DMNDecisionRule(
            rule_id="M6",
            input_conditions={
                "order_quantity": ">6",
                "delivery_date": "(30,45]",
                "inspection_result": "Accept",
                "machine_status": "Well"
            },
            output_value="Multiple Lines",
            description="Large order, normal delivery: multiple lines production"
        ))
        self.add_rule(DMNDecisionRule(
            rule_id="M7",
            input_conditions={
                "order_quantity": ">6",
                "delivery_date": ">45",
                "inspection_result": "Accept",
                "machine_status": "Well"
            },
            output_value="Outsourced",
            description="Large order, loose delivery: outsourced production"
        ))
        # Environment anomaly rule: machine fault triggers outsourced production
        self.add_rule(DMNDecisionRule(
            rule_id="M8",
            input_conditions={"machine_status": "Faults"},
            output_value="Outsourced",
            description="Machine fault: force outsourced production"
        ))
        # Material rejection rule
        self.add_rule(DMNDecisionRule(
            rule_id="M9",
            input_conditions={"inspection_result": "Reject"},
            output_value="Outsourced",
            description="Material rejected: force outsourced production"
        ))

# --------------------------
# SAC Compliance Checker (Definition 13 in Section 5.3)
# --------------------------
class SACChecker:
    """
    Decision Service Adherence Criterion (SAC) Checker
    Aligned with Definition 13 in Section 5.3, for integrated model consistency analysis
    """
    @staticmethod
    def check_sac_compliance(decision_model: DMNDecisionModel, actual_inputs: Set[str]) -> Tuple[bool, Set[str]]:
        """
        Check if the actual inputs satisfy the SAC requirement
        :param decision_model: Target DMN decision model
        :param actual_inputs: Set of actual input attributes provided by the process
        :return: (is_sac_compliant, missing_required_inputs)
        """
        missing_inputs = decision_model.required_inputs - actual_inputs
        is_compliant = len(missing_inputs) == 0
        return is_compliant, missing_inputs

    @staticmethod
    def classify_impact_type(
        decision_model: DMNDecisionModel,
        original_inputs: Dict,
        updated_inputs: Dict,
        original_output: Optional[any],
        new_output: Optional[any],
        execution_success: bool,
        downstream_decisions: Set[str]
    ) -> Tuple[int, str]:
        """
        Classify change impact type into 4 categories (Table 4 in Section 5.3)
        :return: (impact_type_id, impact_type_description)
        """
        actual_input_set = set(updated_inputs.keys())
        sac_compliant, _ = SACChecker.check_sac_compliance(decision_model, actual_input_set)

        # Impact Type 3: Decision Interface Mismatch (SAC not satisfied)
        if not sac_compliant:
            return 3, "Decision interface mismatch, inconsistency at interface level"
        
        # Impact Type 2: Decision Rule Coverage Insufficient (SAC satisfied, no valid output)
        if sac_compliant and not execution_success:
            return 2, "Decision rule coverage insufficient, inconsistency at rule level"
        
        # Impact Type 4: Cross-Decision Propagation (output changed, affects downstream decisions)
        if sac_compliant and execution_success and new_output != original_output and len(downstream_decisions) > 0:
            return 4, "Cross-decision propagation, inconsistency may spread to downstream decisions"
        
        # Impact Type 1: Only affects process execution
        return 1, "Only affects process execution, integrated model remains consistent"
