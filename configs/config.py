Python 3.9.0 (tags/v3.9.0:9cf6752, Oct  5 2020, 15:34:40) [MSC v.1927 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>>"""
configs/config.py
Global experiment configuration, fully matched with paper Table 7 and Section 6.1
All parameters are strictly consistent with the paper
"""
# Fixed random seed for full reproducibility
SEED = 2024

# Global experiment configuration (matches paper Table 7)
EXPERIMENT_CONFIG = {
    # Change injection settings
    "injection_ratios": [0.05, 0.1, 0.2],  # 5%, 10%, 20%
    "injections_per_trace": 1,  # single-shot injection per trace
    "injection_point_range": (0.3, 0.7),  # inject in middle 30%-70% of trace
    "abnormal_coverage_K": 3,  # K=3, abnormal state covers next 3 events
    # Environmental data preprocessing settings (paper Section 4.1)
    "window_length_minutes": 5,  # 5 minutes sliding window
    "continuous_threshold_quantile": 0.9,  # 0.9 quantile for abnormal threshold
    "default_env_state_normal": "normal",
    "default_env_state_abnormal": "abnormal",
    # Algorithm parameters
    "ccpa_bfs_max_depth": None,  # no max depth, full propagation
    "dmn_decision_table_path": "models/dmn/manufacturing_decision.dmn",
    "bpmn_process_model_path": "models/bpmn/manufacturing_process.bpmn",
    # Train/test split for learning-based baselines
    "train_test_split": 0.7,
    "cross_validation_folds": 5,
    # GPU settings for deep learning baselines
    "use_gpu": True,
    "batch_size": 64,
    "lstm_hidden_dim": 64,
    "lstm_num_layers": 2,
    "training_epochs": 50,
    "learning_rate": 1e-3,
}

# Dataset configuration (matches paper Table 5 and Table 6)
DATASET_CONFIG = {
    "RTF": {
        "name": "Road Traffic Fine Management",
        "file_name": "rtf.xes.gz",
        "doi": "https://doi.org/10.4121/uuid:270fd440-1057-4fb9-89a9-b699b47990f5",
        "download_url": "https://data.4tu.nl/file/270fd440-1057-4fb9-89a9-b699b47990f5/1f05d408-ee39-439c-9131-0e44978a304c",
        "target_attributes": ["amount", "points"],  # Attributes for change injection
        "environmental_variables": [
            {
                "name": "workload",
                "type": "discrete",
                "normal_condition": "normal",
                "abnormal_condition": "high",
                "logged_field": "env_workload_state"
            },
            {
                "name": "system_delay",
                "type": "continuous",
                "logged_field": "env_delay_state"
            }
        ]
    },
    "Sepsis": {
        "name": "Sepsis Cases Treatment Process",
        "file_name": "sepsis.xes.gz",
        "doi": "https://doi.org/10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460",
        "download_url": "https://data.4tu.nl/file/915d2bfb-7e84-49ad-a286-dc35f063a460/8a9d8d09-9c0c-400f-9007-0e44978a304c",
        "target_attributes": ["CRP", "LacticAcid", "Leucocytes"],  # Attributes for change injection
        "environmental_variables": [
            {
                "name": "bed_occupancy",
                "type": "continuous",
                "logged_field": "env_bed_state"
            },
            {
                "name": "staff_level",
                "type": "discrete",
                "normal_condition": "adequate",
                "abnormal_condition": "shortage",
                "logged_field": "env_staff_state"
            }
        ]
    },
    "BPIC2012": {
        "name": "BPIC2012 Financial Loan Application",
        "file_name": "bpic2012.xes.gz",
        "doi": "https://doi.org/10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f",
        "download_url": "https://data.4tu.nl/file/3926db30-f712-4394-aebc-75976070e91f/3f3a5c50-0d89-4e3f-8c07-0e44978a304c",
        "target_attributes": ["AMOUNT_REQ"],  # Attributes for change injection
        "environmental_variables": [
            {
                "name": "credit_risk_level",
                "type": "discrete",
                "normal_condition": "low",
                "abnormal_condition": "high",
                "logged_field": "env_risk_state"
            },
            {
                "name": "service_outage",
                "type": "discrete",
                "normal_condition": "up",
                "abnormal_condition": "down",
                "logged_field": "env_service_state"
            }
        ]
    },
    "Synthetic": {
        "name": "Smart Manufacturing Synthetic Process",
        "file_name": "synthetic.xes",
        "target_attributes": ["order_quantity", "delivery_date"],  # Attributes for change injection
        "process_model_path": "models/bpmn/manufacturing_process.bpmn",
        "plg2_simulation_settings": {
            "num_traces": 10000,
            "num_events": 20000,
            "random_seed": SEED
        },
        "environmental_variables": [
            {
                "name": "temp",
                "type": "continuous",
                "logged_field": "env_temp_state"
            },
            {
                "name": "material_supply",
                "type": "discrete",
                "normal_condition": "sufficient",
                "abnormal_condition": "shortage",
                "logged_field": "env_supply_state"
            }
        ]
    }
}
