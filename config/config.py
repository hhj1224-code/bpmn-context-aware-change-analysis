Python 3.9.0 (tags/v3.9.0:9cf6752, Oct  5 2020, 15:34:40) [MSC v.1927 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>>### `configs/experiment_config.py`
```python
# Corresponding Section: 6.2 Experimental Setup, Table 7 Injection settings and key parameters
import numpy as np

# Fixed random seed for full reproducibility
RANDOM_SEED = 2024
np.random.seed(RANDOM_SEED)

# Global experimental configuration
GLOBAL_CONFIG = {
    "injection_ratios": [0.05, 0.10, 0.20],
    "injections_per_trace": 1,
    "injection_point_range": (0.3, 0.7),
    "abnormal_coverage_k": 3,
    "window_length_minutes": 5,
    "continuous_threshold_quantile": 0.9,
    "k_fold": 5,
    "train_test_split": 0.7,
    "datasets": ["RTF", "Sepsis", "BPIC2012", "Synthetic"],
    "scenarios": ["S1", "S2", "S3"]
}

# Environmental variable configuration for each dataset (Table 6)
DATASET_ENV_CONFIG = {
    "RTF": [
        {
            "name": "workload",
            "type": "discrete",
            "mapping": {"normal": "normal", "high": "abnormal"},
            "logged_field": "env_workload_state"
        },
        {
            "name": "system_delay",
            "type": "continuous",
            "logged_field": "env_delay_state"
        }
    ],
    "Sepsis": [
        {
            "name": "bed_occupancy",
            "type": "continuous",
            "logged_field": "env_bed_state"
        },
        {
            "name": "staff_level",
            "type": "discrete",
            "mapping": {"adequate": "normal", "shortage": "abnormal"},
            "logged_field": "env_staff_state"
        }
    ],
    "BPIC2012": [
        {
            "name": "credit_risk_level",
            "type": "discrete",
            "mapping": {"low": "normal", "high": "abnormal"},
            "logged_field": "env_risk_state"
        },
        {
            "name": "service_outage",
            "type": "discrete",
            "mapping": {"up": "normal", "down": "abnormal"},
            "logged_field": "env_service_state"
        }
    ],
    "Synthetic": [
        {
            "name": "temp",
            "type": "continuous",
            "logged_field": "env_temp_state"
        },
        {
            "name": "material_supply",
            "type": "discrete",
            "mapping": {"sufficient": "normal", "shortage": "abnormal"},
            "logged_field": "env_supply_state"
        }
    ]
}

# Injection attributes for each dataset
DATASET_INJECT_ATTRS = {
    "RTF": ["amount", "points"],
    "Sepsis": ["CRP", "LacticAcid", "Leucocytes"],
    "BPIC2012": ["AMOUNT_REQ"],
    "Synthetic": ["order_quantity", "delivery_date"]
}
