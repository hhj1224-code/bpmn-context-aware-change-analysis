Python 3.9.0 (tags/v3.9.0:9cf6752, Oct  5 2020, 15:34:40) [MSC v.1927 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> """
utils/__init__.py
Initialize utils module
"""
from utils.preprocessing_utils import *
from utils.metrics import *"""
utils/preprocessing_utils.py
Context data preprocessing functions, corresponds to paper Section 4.1
Implements sliding window preprocessing, semantic discretization, missing value handling
"""
import pandas as pd
import numpy as np
from pm4py.objects.log.obj import EventLog, Trace
from tqdm import tqdm
from configs.config import EXPERIMENT_CONFIG, DATASET_CONFIG

def load_event_log(dataset_name, synthetic=False, config=EXPERIMENT_CONFIG):
    """
    Load event log from xes file
    Args:
        dataset_name: Name of the dataset
        synthetic: Whether it is synthetic dataset
        config: Global experiment config
    Returns:
        event_log: PM4Py EventLog object
        process_model: BPMN process model object (for synthetic dataset)
    """
    import pm4py
    dataset_config = DATASET_CONFIG[dataset_name]
    
    if synthetic:
        # Generate synthetic log using PLG2 simulation
        from pm4py.algo.simulation.playout.process_tree import algorithm as tree_playout
        from pm4py.objects.process_tree.utils.generic import parse as pt_parse
        
        # Load BPMN model and convert to process tree for simulation
        bpmn_model = pm4py.read_bpmn(dataset_config["process_model_path"])
        process_tree = pm4py.convert_to_process_tree(bpmn_model)
        
        # Playout process tree to generate synthetic log
        event_log = tree_playout.apply(
            process_tree,
            parameters={
                "num_traces": dataset_config["plg2_simulation_settings"]["num_traces"],
                "random_seed": dataset_config["plg2_simulation_settings"]["random_seed"]
            }
        )
        return event_log, bpmn_model
    else:
        # Load real event log from xes file
        file_path = f"data/raw/{dataset_config['file_name']}"
        event_log = pm4py.read_xes(file_path)
        # Discover process model using Inductive Miner
        process_model, initial_marking, final_marking = pm4py.discover_bpmn_inductive(event_log)
        return event_log, process_model

def sliding_window_preprocessing(observations, timestamps, window_length_minutes, variable_type):
    """
    Sliding window preprocessing for environmental time series, paper Section 4.1
    Args:
        observations: List of raw environmental observations
        timestamps: List of corresponding timestamps
        window_length_minutes: Length of sliding window in minutes
        variable_type: Type of variable, 'continuous' or 'discrete'
    Returns:
        processed_observations: Preprocessed observations aligned with timestamps
    """
    window_length = pd.Timedelta(minutes=window_length_minutes)
    processed_observations = []
    
    for i, current_ts in enumerate(timestamps):
        # Get window start time
        window_start = current_ts - window_length
        # Get all observations in the window
        window_mask = (timestamps >= window_start) & (timestamps <= current_ts)
        window_data = observations[window_mask]
        
        if len(window_data) == 0:
            # No data in window, use last valid observation
            processed_val = processed_observations[-1] if len(processed_observations) > 0 else np.nan
        else:
            if variable_type == "continuous":
                # Use median for continuous variables to suppress high-frequency noise
                processed_val = np.median(window_data)
            else:
                # Use last valid observation (LOCF) for discrete variables
                processed_val = window_data[-1]
        
        processed_observations.append(processed_val)
    
    # Handle missing values with linear interpolation for continuous variables
    processed_observations = pd.Series(processed_observations)
    if variable_type == "continuous":
        processed_observations = processed_observations.interpolate(method="linear")
    else:
        processed_observations = processed_observations.ffill().bfill()
    
    return processed_observations.values

def semantic_discretization(processed_values, variable_config, config=EXPERIMENT_CONFIG):
    """
    Semantic discretization of environmental data, paper Section 4.1
    Maps continuous numerical values to discrete semantic states (normal/abnormal)
    Args:
        processed_values: Preprocessed numerical values
        variable_config: Configuration of the environmental variable
        config: Global experiment config
    Returns:
        discrete_states: Discrete semantic states for each observation
    """
    if variable_config["type"] == "discrete":
        # Direct mapping for discrete variables
        normal_condition = variable_config["normal_condition"]
        discrete_states = np.where(
            processed_values == normal_condition,
            config["default_env_state_normal"],
            config["default_env_state_abnormal"]
        )
    else:
        # Continuous variables: use 0.9 quantile as threshold
        threshold = np.quantile(processed_values[~np.isnan(processed_values)], config["continuous_threshold_quantile"])
        discrete_states = np.where(
            processed_values <= threshold,
            config["default_env_state_normal"],
            config["default_env_state_abnormal"]
        )
    return discrete_states

def preprocess_environmental_data(event_log, dataset_name, config=EXPERIMENT_CONFIG):
    """
    Full preprocessing pipeline for environmental data, paper Section 4.1
    Args:
        event_log: Raw PM4Py EventLog object
        dataset_name: Name of the dataset
        config: Global experiment config
    Returns:
        enhanced_log: EventLog with preprocessed environmental context states
    """
    dataset_config = DATASET_CONFIG[dataset_name]
    env_variables = dataset_config["environmental_variables"]
    window_length = config["window_length_minutes"]
    
    # Collect all timestamps and event data from the log
    all_timestamps = []
    for trace in event_log:
        for event in trace:
            all_timestamps.append(event["time:timestamp"])
    all_timestamps = pd.to_datetime(all_timestamps)
    
    # Generate synthetic environmental data for real datasets (aligned with event timestamps)
    np.random.seed(config["random_seed"] if "random_seed" in config else 2024)
    env_data_dict = {}
    
    for env_var in env_variables:
        var_name = env_var["name"]
        var_type = env_var["type"]
        
        # Generate synthetic raw observations
        if var_type == "continuous":
            raw_observations = np.random.normal(loc=25, scale=5, size=len(all_timestamps))
        else:
            categories = [env_var["normal_condition"], env_var["abnormal_condition"]]
            raw_observations = np.random.choice(categories, size=len(all_timestamps), p=[0.9, 0.1])
        
        # Sliding window preprocessing
        processed_values = sliding_window_preprocessing(
            raw_observations, all_timestamps, window_length, var_type
        )
        
        # Semantic discretization
        discrete_states = semantic_discretization(processed_values, env_var, config)
        
        # Store in dict
        env_data_dict[env_var["logged_field"]] = discrete_states
    
    # Add environmental states to event log
    enhanced_log = EventLog()
    event_idx = 0
    
    for trace in tqdm(event_log, desc=f"Preprocessing {dataset_name} environmental data"):
        new_trace = Trace(attributes=trace.attributes)
        for event in trace:
            new_event = event.copy()
            # Add all environmental states to the event
            for logged_field, states in env_data_dict.items():
                new_event[logged_field] = states[event_idx]
            new_trace.append(new_event)
            event_idx += 1
        enhanced_log.append(new_trace)
    
    return enhanced_log

def generate_context_enhanced_logs(preprocessed_log, dataset_name, config=EXPERIMENT_CONFIG):
    """
    Generate context-enhanced event logs with internal and external context
    Args:
        preprocessed_log: Preprocessed event log with environmental data
        dataset_name: Name of the dataset
        config: Global experiment config
    Returns:
        context_enhanced_log: EventLog with full context information
    """
    context_enhanced_log = EventLog()
    
    for trace in tqdm(preprocessed_log, desc=f"Generating context-enhanced log for {dataset_name}"):
        new_trace = Trace(attributes=trace.attributes)
        # Track internal context: behavior context (executed activities) and data context
        behavior_context = []
        data_context = {}
        
        for event in trace:
            new_event = event.copy()
            # Update behavior context
            behavior_context.append(event["concept:name"])
            new_event["internal_behavior_context"] = "|".join(behavior_context)
            
            # Update data context (all event attributes except standard ones)
            standard_attrs = ["concept:name", "time:timestamp", "lifecycle:transition", "org:resource"]
            for attr_key, attr_value in event.items():
                if attr_key not in standard_attrs and not attr_key.startswith("env_"):
                    data_context[attr_key] = attr_value
            new_event["internal_data_context"] = str(data_context)
            
            # Collect external context (environmental states)
            external_context = {}
            for attr_key, attr_value in event.items():
                if attr_key.startswith("env_"):
                    external_context[attr_key] = attr_value
            new_event["external_context"] = str(external_context)
            
            new_trace.append(new_event)
        context_enhanced_log.append(new_trace)
    
    return context_enhanced_log