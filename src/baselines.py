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
    return 1"""
src/baselines.py
4 baseline methods full implementation, corresponds to paper Section 6.2
Implements:
1. DIS: Data Impact Analysis (Tsoury et al. 2020) [2]
2. DTM: Decision Tree Model (Rozinat et al. 2006) [45]
3. LSTM-PPM: LSTM-based Predictive Process Monitoring (Tax et al. 2017) [46]
4. BINet: Multi-perspective Business Process Anomaly Classification (Nolle et al. 2022) [47]
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from configs.config import EXPERIMENT_CONFIG, SEED

# Fix random seed
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ==================== Baseline 1: DIS Data Impact Analysis ====================
class DIS_Baseline:
    """
    Data Impact Analysis (DIS) baseline, Tsoury et al. 2020 [2]
    Converts process model to relational database representation, uses SQL queries to retrieve affected elements
    """
    def __init__(self, config=EXPERIMENT_CONFIG):
        self.config = config
        self.data_dependencies = None
    
    def fit(self, event_log, ground_truth):
        """
        Build data dependency map from event log
        """
        # Extract data dependencies between activities
        self.data_dependencies = {}
        activity_data_map = {}
        
        for trace in event_log:
            for event_idx, event in enumerate(trace):
                activity = event["concept:name"]
                standard_attrs = ["concept:name", "time:timestamp", "lifecycle:transition", "org:resource"]
                data_attrs = {k: v for k, v in event.items() if k not in standard_attrs}
                
                if activity not in activity_data_map:
                    activity_data_map[activity] = {"input": set(), "output": set()}
                
                # Output data: attributes written by this activity
                for attr in data_attrs.keys():
                    activity_data_map[activity]["output"].add(attr)
                
                # Input data: attributes from previous events
                if event_idx > 0:
                    prev_event = trace[event_idx - 1]
                    prev_attrs = {k: v for k, v in prev_event.items() if k not in standard_attrs}
                    for attr in prev_attrs.keys():
                        if attr in data_attrs:
                            activity_data_map[activity]["input"].add(attr)
        
        # Build data dependency graph
        self.data_dependencies = activity_data_map
        return self
    
    def predict(self, event_log):
        """
        Predict affected traces using data dependency retrieval
        """
        y_pred = []
        
        for trace in event_log:
            changed_data = set()
            previous_data = {}
            
            # Detect changed data attributes
            for event in trace:
                standard_attrs = ["concept:name", "time:timestamp", "lifecycle:transition", "org:resource"]
                data_attrs = {k: v for k, v in event.items() if k not in standard_attrs}
                
                for attr, value in data_attrs.items():
                    if attr in previous_data:
                        if previous_data[attr] != value:
                            changed_data.add(attr)
                    previous_data[attr] = value
            
            # Retrieve affected activities
            affected_activities = set()
            for activity, io_data in self.data_dependencies.items():
                input_data = io_data["input"]
                if len(changed_data & input_data) > 0:
                    affected_activities.add(activity)
            
            # Predict 1 if any affected activities, else 0
            y_pred.append(1 if len(affected_activities) > 0 else 0)
        
        return np.array(y_pred)

# ==================== Baseline 2: DTM Decision Tree Model ====================
class DTM_Baseline:
    """
    Decision Tree Model (DTM) baseline, Rozinat et al. 2006 [45]
    Binary classification of affected traces using case-level features
    """
    def __init__(self, config=EXPERIMENT_CONFIG):
        self.config = config
        self.model = DecisionTreeClassifier(random_state=SEED)
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def _extract_features(self, event_log):
        """
        Extract case-level features from event log
        """
        features = []
        for trace in event_log:
            trace_features = {}
            # Trace length
            trace_features["trace_length"] = len(trace)
            # Number of unique activities
            trace_features["unique_activities"] = len({event["concept:name"] for event in trace})
            # Number of data attribute changes
            changed_data_count = 0
            previous_data = {}
            for event in trace:
                standard_attrs = ["concept:name", "time:timestamp", "lifecycle:transition", "org:resource"]
                data_attrs = {k: v for k, v in event.items() if k not in standard_attrs}
                for attr, value in data_attrs.items():
                    if attr in previous_data:
                        if previous_data[attr] != value:
                            changed_data_count += 1
                    previous_data[attr] = value
            trace_features["changed_data_count"] = changed_data_count
            # Environmental state changes
            env_change_count = 0
            previous_env = {}
            for event in trace:
                env_attrs = {k: v for k, v in event.items() if k.startswith("env_")}
                for attr, value in env_attrs.items():
                    if attr in previous_env:
                        if previous_env[attr] != value:
                            env_change_count += 1
                    previous_env[attr] = value
            trace_features["env_change_count"] = env_change_count
            
            features.append(trace_features)
        
        return pd.DataFrame(features)
    
    def fit(self, event_log, ground_truth):
        """
        Train decision tree classifier
        """
        # Extract features
        X = self._extract_features(event_log)
        self.feature_names = X.columns
        
        # Extract labels
        y = []
        for trace in event_log:
            trace_id = trace.attributes["concept:name"] if "concept:name" in trace.attributes else f"trace_{len(y)}"
            y.append(ground_truth[trace_id]["label"])
        y = np.array(y)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=self.config["train_test_split"], random_state=SEED, stratify=y if len(np.unique(y))>1 else None
        )
        
        # Preprocess features
        X_train = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, event_log):
        """
        Predict affected traces
        """
        X = self._extract_features(event_log)
        X = self.scaler.transform(X)
        y_pred = self.model.predict(X)
        return y_pred

# ==================== Baseline 3: LSTM-PPM Predictive Process Monitoring ====================
class ProcessDataset(Dataset):
    def __init__(self, sequences, labels=None, max_seq_len=100):
        self.sequences = sequences
        self.labels = labels
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # Pad sequence to max length
        if len(seq) < self.max_seq_len:
            seq = np.pad(seq, ((0, self.max_seq_len - len(seq)), (0, 0)), mode="constant")
        else:
            seq = seq[:self.max_seq_len]
        
        seq_tensor = torch.tensor(seq, dtype=torch.float32)
        if self.labels is not None:
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
            return seq_tensor, label_tensor
        return seq_tensor

class LSTM_PPM_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes=2):
        super(LSTM_PPM_Model, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        # Use last hidden state
        out = self.dropout(hidden[-1, :, :])
        out = self.fc(out)
        return out

class LSTM_PPM_Baseline:
    """
    LSTM-based Predictive Process Monitoring (LSTM-PPM) baseline, Tax et al. 2017 [46]
    Binary classification of affected traces using LSTM on event sequences
    """
    def __init__(self, config=EXPERIMENT_CONFIG):
        self.config = config
        self.model = None
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.scaler = StandardScaler()
        self.activity_vocab = None
        self.max_seq_len = 100
        self.device = torch.device("cuda" if config["use_gpu"] and torch.cuda.is_available() else "cpu")
    
    def _encode_sequence(self, trace):
        """
        Encode trace into sequence of vectors
        """
        sequence = []
        for event in trace:
            event_vector = []
            # One-hot encode activity
            activity = event["concept:name"]
            activity_onehot = np.zeros(len(self.activity_vocab))
            if activity in self.activity_vocab:
                activity_onehot[self.activity_vocab[activity]] = 1
            event_vector.extend(activity_onehot)
            
            # Add numerical data attributes
            standard_attrs = ["concept:name", "time:timestamp", "lifecycle:transition", "org:resource"]
            data_attrs = {k: v for k, v in event.items() if k not in standard_attrs and not k.startswith("env_")}
            numerical_values = []
            for attr in sorted(data_attrs.keys()):
                val = data_attrs[attr]
                numerical_values.append(float(val) if isinstance(val, (int, float)) else 0.0)
            event_vector.extend(numerical_values)
            
            # Add environmental states
            env_attrs = {k: v for k, v in event.items() if k.startswith("env_")}
            env_values = []
            for attr in sorted(env_attrs.keys()):
                env_values.append(1.0 if env_attrs[attr] == self.config["default_env_state_abnormal"] else 0.0)
            event_vector.extend(env_values)
            
            sequence.append(event_vector)
        
        return np.array(sequence)
    
    def fit(self, event_log, ground_truth):
        """
        Train LSTM model
        """
        # Build activity vocabulary
        self.activity_vocab = {}
        for trace in event_log:
            for event in trace:
                activity = event["concept:name"]
                if activity not in self.activity_vocab:
                    self.activity_vocab[activity] = len(self.activity_vocab)
        
        # Encode all traces into sequences
        sequences = []
        for trace in tqdm(event_log, desc="Encoding sequences for LSTM-PPM"):
            sequences.append(self._encode_sequence(trace))
        
        # Extract labels
        y = []
        for trace in event_log:
            trace_id = trace.attributes["concept:name"] if "concept:name" in trace.attributes else f"trace_{len(y)}"
            y.append(ground_truth[trace_id]["label"])
        y = np.array(y)
        
        # Train/test split
        train_idx, test_idx = train_test_split(
            np.arange(len(sequences)), train_size=self.config["train_test_split"], random_state=SEED, stratify=y if len(np.unique(y))>1 else None
        )
        train_sequences = [sequences[i] for i in train_idx]
        train_labels = y[train_idx]
        test_sequences = [sequences[i] for i in test_idx]
        test_labels = y[test_idx]
        
        # Create datasets and dataloaders
        input_dim = len(train_sequences[0][0]) if len(train_sequences) > 0 and len(train_sequences[0]) > 0 else 10
        train_dataset = ProcessDataset(train_sequences, train_labels, self.max_seq_len)
        test_dataset = ProcessDataset(test_sequences, test_labels, self.max_seq_len)
        train_loader = DataLoader(train_dataset, batch_size=self.config["batch_size"], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config["batch_size"], shuffle=False)
        
        # Initialize model
        self.model = LSTM_PPM_Model(
            input_dim=input_dim,
            hidden_dim=self.config["lstm_hidden_dim"],
            num_layers=self.config["lstm_num_layers"],
            num_classes=2
        ).to(self.device)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])
        
        # Train model
        self.model.train()
        for epoch in range(self.config["training_epochs"]):
            train_loss = 0.0
            for batch_seq, batch_label in train_loader:
                batch_seq = batch_seq.to(self.device)
                batch_label = batch_label.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_seq)
                loss = criterion(outputs, batch_label)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_seq, batch_label in test_loader:
                    batch_seq = batch_seq.to(self.device)
                    batch_label = batch_label.to(self.device)
                    outputs = self.model(batch_seq)
                    loss = criterion(outputs, batch_label)
                    val_loss += loss.item()
            self.model.train()
        
        return self
    
    def predict(self, event_log):
        """
        Predict affected traces
        """
        self.model.eval()
        # Encode traces
        sequences = []
        for trace in tqdm(event_log, desc="Predicting with LSTM-PPM"):
            sequences.append(self._encode_sequence(trace))
        
        # Create dataset
        dataset = ProcessDataset(sequences, max_seq_len=self.max_seq_len)
        loader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=False)
        
        # Predict
        y_pred = []
        with torch.no_grad():
            for batch_seq in loader:
                batch_seq = batch_seq.to(self.device)
                outputs = self.model(batch_seq)
                preds = torch.argmax(outputs, dim=1)
                y_pred.extend(preds.cpu().numpy())
        
        return np.array(y_pred)

# ==================== Baseline 4: BINet Anomaly Detection ====================
class BINet_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(BINet_Model, self).__init__()
        self.lstm_encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm_decoder = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True)
        self.hidden_dim = hidden_dim
    
    def forward(self, x):
        # Encode
        encoder_out, (hidden, cell) = self.lstm_encoder(x)
        # Decode
        decoder_out, _ = self.lstm_decoder(encoder_out, (hidden, cell))
        return decoder_out

class BINet_Baseline:
    """
    BINet: Multi-perspective Business Process Anomaly Classification baseline, Nolle et al. 2022 [47]
    Anomaly detection using LSTM autoencoder
    """
    def __init__(self, config=EXPERIMENT_CONFIG):
        self.config = config
        self.model = None
        self.activity_vocab = None
        self.max_seq_len = 100
        self.threshold = 0.5
        self.device = torch.device("cuda" if config["use_gpu"] and torch.cuda.is_available() else "cpu")
    
    def _encode_sequence(self, trace):
        """
        Encode trace into sequence of vectors
        """
        sequence = []
        for event in trace:
            event_vector = []
            # One-hot encode activity
            activity = event["concept:name"]
            activity_onehot = np.zeros(len(self.activity_vocab))
            if activity in self.activity_vocab:
                activity_onehot[self.activity_vocab[activity]] = 1
            event_vector.extend(activity_onehot)
            
            # Add numerical data attributes
            standard_attrs = ["concept:name", "time:timestamp", "lifecycle:transition", "org:resource"]
            data_attrs = {k: v for k, v in event.items() if k not in standard_attrs and not k.startswith("env_")}
            numerical_values = []
            for attr in sorted(data_attrs.keys()):
                val = data_attrs[attr]
                numerical_values.append(float(val) if isinstance(val, (int, float)) else 0.0)
            event_vector.extend(numerical_values)
            
            # Add environmental states
            env_attrs = {k: v for k, v in event.items() if k.startswith("env_")}
            env_values = []
            for attr in sorted(env_attrs.keys()):
                env_values.append(1.0 if env_attrs[attr] == self.config["default_env_state_abnormal"] else 0.0)
            event_vector.extend(env_values)
            
            sequence.append(event_vector)
        
        return np.array(sequence)
    
    def fit(self, event_log, ground_truth):
        """
        Train BINet autoencoder on normal (unaffected) traces
        """
        # Build activity vocabulary
        self.activity_vocab = {}
        for trace in event_log:
            for event in trace:
                activity = event["concept:name"]
                if activity not in self.activity_vocab:
                    self.activity_vocab[activity] = len(self.activity_vocab)
        
        # Separate normal traces for training
        normal_traces = []
        for trace in event_log:
            trace_id = trace.attributes["concept:name"] if "concept:name" in trace.attributes else f"trace_{len(normal_traces)}"
            if ground_truth[trace_id]["label"] == 0:
                normal_traces.append(trace)
        
        # Encode normal traces
        sequences = []
        for trace in tqdm(normal_traces, desc="Encoding sequences for BINet"):
            sequences.append(self._encode_sequence(trace))
        
        # Train/test split
        train_idx, test_idx = train_test_split(
            np.arange(len(sequences)), train_size=self.config["train_test_split"], random_state=SEED
        )
        train_sequences = [sequences[i] for i in train_idx]
        test_sequences = [sequences[i] for i in test_idx]
        
        # Create datasets and dataloaders
        input_dim = len(train_sequences[0][0]) if len(train_sequences) > 0 and len(train_sequences[0]) > 0 else 10
        train_dataset = ProcessDataset(train_sequences, max_seq_len=self.max_seq_len)
        test_dataset = ProcessDataset(test_sequences, max_seq_len=self.max_seq_len)
        train_loader = DataLoader(train_dataset, batch_size=self.config["batch_size"], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config["batch_size"], shuffle=False)
        
        # Initialize model
        self.model = BINet_Model(
            input_dim=input_dim,
            hidden_dim=self.config["lstm_hidden_dim"],
            num_layers=self.config["lstm_num_layers"]
        ).to(self.device)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])
        
        # Train model
        self.model.train()
        for epoch in range(self.config["training_epochs"]):
            train_loss = 0.0
            for batch_seq in train_loader:
                batch_seq = batch_seq.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_seq)
                loss = criterion(outputs, batch_seq)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
        
        # Calculate threshold on test set
        self.model.eval()
        reconstruction_errors = []
        with torch.no_grad():
            for batch_seq in test_loader:
                batch_seq = batch_seq.to(self.device)
                outputs = self.model(batch_seq)
                mse = torch.mean((outputs - batch_seq) ** 2, dim=(1, 2))
                reconstruction_errors.extend(mse.cpu().numpy())
        
        # Set threshold to 0.9 quantile of reconstruction errors
        self.threshold = np.quantile(reconstruction_errors, 0.9)
        return self
    
    def predict(self, event_log):
        """
        Predict affected traces (anomalies)
        """
        self.model.eval()
        # Encode traces
        sequences = []
        for trace in tqdm(event_log, desc="Predicting with BINet"):
            sequences.append(self._encode_sequence(trace))
        
        # Create dataset
        dataset = ProcessDataset(sequences, max_seq_len=self.max_seq_len)
        loader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=False)
        
        # Predict
        y_pred = []
        with torch.no_grad():
            for batch_seq in loader:
                batch_seq = batch_seq.to(self.device)
                outputs = self.model(batch_seq)
                mse = torch.mean((outputs - batch_seq) ** 2, dim=(1, 2))
                preds = (mse > self.threshold).cpu().numpy().astype(int)
                y_pred.extend(preds)
        
        return np.array(y_pred)