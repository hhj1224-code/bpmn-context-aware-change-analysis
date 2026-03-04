Python 3.9.0 (tags/v3.9.0:9cf6752, Oct  5 2020, 15:34:40) [MSC v.1927 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> """
models/bpmn_executor.py
BPMN process execution engine for synthetic process simulation
Corresponds to paper Fig.2 smart manufacturing process model
"""
import pm4py
from pm4py.objects.bpmn.obj import BPMN
from pm4py.algo.simulation.playout.bpmn import algorithm as bpmn_playout

def load_bpmn_model(file_path):
    """
    Load BPMN model from file
    Args:
        file_path: Path to .bpmn file
    Returns:
        bpmn_model: PM4Py BPMN model object
    """
    return pm4py.read_bpmn(file_path)

def simulate_bpmn_process(bpmn_model, num_traces=10000, random_seed=2024):
    """
    Simulate BPMN process to generate synthetic event log
    Args:
        bpmn_model: PM4Py BPMN model object
        num_traces: Number of traces to generate
        random_seed: Random seed for reproducibility
    Returns:
        event_log: Simulated PM4Py EventLog object
    """
    # Convert BPMN to process tree for playout
    process_tree = pm4py.convert_to_process_tree(bpmn_model)
    
    # Simulate process execution
    event_log = pm4py.playout.process_tree.playout(
        process_tree,
        parameters={
            "num_traces": num_traces,
            "random_seed": random_seed
        }
    )
    
    return event_log

def get_process_data_attributes(bpmn_model, event_log):
    """
    Extract all data attributes from BPMN model and event log
    Args:
        bpmn_model: PM4Py BPMN model object
        event_log: PM4Py EventLog object
    Returns:
        data_attributes: List of all data attributes in the process
    """
    # Extract attributes from event log
    event_attrs = set()
    for trace in event_log:
        for event in trace:
            event_attrs.update(event.keys())
    
    # Standard BPMN attributes to exclude
    standard_attrs = ["concept:name", "time:timestamp", "lifecycle:transition", "org:resource"]
    data_attributes = [attr for attr in event_attrs if attr not in standard_attrs and not attr.startswith("env_")]
    
    return data_attributes