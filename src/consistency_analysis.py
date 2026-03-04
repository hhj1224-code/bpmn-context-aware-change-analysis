Python 3.9.0 (tags/v3.9.0:9cf6752, Oct  5 2020, 15:34:40) [MSC v.1927 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>># Corresponding Section: 5.3 Consistency Analysis and Impact Type Classification, Table 4, Definition 13
from typing import Tuple

def classify_impact_type(
    sac_check_result: bool,
    has_valid_output: bool,
    has_cascade_propagation: bool,
    process_affected: bool,
    decision_affected: bool
) -> int:
    """
    Classifies the change impact on the integrated model into 4 types.
    Matches Table 4 in the paper.

    Parameters
    ----------
    sac_check_result : bool
        Result of SAC consistency check (True = interface consistent)
    has_valid_output : bool
        Whether DMN model produces a valid output for the updated inputs
    has_cascade_propagation : bool
        Whether the output change propagates to subsequent decisions
    process_affected : bool
        Whether the process layer is affected by the change
    decision_affected : bool
        Whether the decision layer is affected by the change

    Returns
    -------
    int
        Impact type ID (1-4), 0 = no impact
    """
    # Type 1: Only affects process execution
    if sac_check_result and has_valid_output and not has_cascade_propagation and process_affected and not decision_affected:
        return 1
    # Type 2: Insufficient decision rule coverage
    elif sac_check_result and not has_valid_output and not has_cascade_propagation and process_affected and decision_affected:
        return 2
    # Type 3: Decision interface mismatch
    elif not sac_check_result and not has_valid_output and not has_cascade_propagation and process_affected and decision_affected:
        return 3
    # Type 4: Cross-decision cascade propagation
    elif sac_check_result and has_valid_output and has_cascade_propagation and process_affected and decision_affected:
        return 4
    # No impact
    else:
        return 0

def get_impact_type_description(impact_type: int) -> str:
    """Returns the textual description of the impact type."""
    impact_descriptions = {
        1: "Type 1: Only affects process execution",
        2: "Type 2: Insufficient decision rule coverage",
        3: "Type 3: Decision interface mismatch",
        4: "Type 4: Cross-decision cascade propagation",
        0: "No impact"
    }
    return impact_descriptions.get(impact_type, "Unknown impact type")
