Python 3.9.0 (tags/v3.9.0:9cf6752, Oct  5 2020, 15:34:40) [MSC v.1927 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> """
src/data_dependency_graph.py
Data Dependency Graph (DDG) construction, corresponds to paper Section 5.1
Implements Definition 11 of the paper, builds directed graph for change propagation
"""
import networkx as nx
from pm4py.objects.log.obj import EventLog
from pm4py.objects.bpmn.obj import BPMN
from tqdm import tqdm
from configs.config import EXPERIMENT_CONFIG

class DataDependencyGraph:
    """
    Data Dependency Graph (DDG) class, matches paper Definition 11
    """
    def __init__(self):
        # Directed graph for DDG
        self.graph = nx.DiGraph()
        # Node sets
        self.activity_nodes = set()
        self.data_nodes = set()
        self.path_constraint_nodes = set()
        # Edge type mapping
        self.edge_types = {}
    
    def add_node(self, node_id, node_type, attributes=None):
        """
        Add node to DDG
        Args:
            node_id: Unique ID of the node
            node_type: Type of node, 'activity', 'data', 'path_constraint'
            attributes: Additional attributes of the node
        """
        self.graph.add_node(node_id, node_type=node_type, attributes=attributes or {})
        
        if node_type == "activity":
            self.activity_nodes.add(node_id)
        elif node_type == "data":
            self.data_nodes.add(node_id)
        elif node_type == "path_constraint":
            self.path_constraint_nodes.add(node_id)
    
    def add_edge(self, source_id, target_id, edge_type, attributes=None):
        """
        Add directed edge to DDG, matches paper Definition 11 edge mapping rules
        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Type of edge, matches paper Table 3
            attributes: Additional attributes of the edge
        """
        self.graph.add_edge(source_id, target_id, edge_type=edge_type, attributes=attributes or {})
        self.edge_types[(source_id, target_id)] = edge_type
    
    def get_successors(self, node_id):
        """
        Get all successor nodes of a given node
        Args:
            node_id: Node ID
        Returns:
            successors: List of (successor_id, edge_type) tuples
        """
        successors = []
        for succ in self.graph.successors(node_id):
            edge_type = self.edge_types[(node_id, succ)]
            successors.append((succ, edge_type))
        return successors
    
    def get_node_type(self, node_id):
        """
        Get type of a node
        Args:
            node_id: Node ID
        Returns:
            node_type: Type of the node
        """
        return self.graph.nodes[node_id]["node_type"]

def build_data_dependency_graph(process_model, event_log, config=EXPERIMENT_CONFIG):
    """
    Build Data Dependency Graph (DDG) from process model and event log
    Corresponds to paper Section 5.1 Definition 11
    Args:
        process_model: BPMN process model object
        event_log: EventLog object
        config: Global experiment config
    Returns:
        ddg: DataDependencyGraph object
    """
    ddg = DataDependencyGraph()
    
    # Step 1: Extract all data attributes from event log
    data_attributes = set()
    activity_names = set()
    for trace in event_log:
        for event in trace:
            activity_names.add(event["concept:name"])
            for attr_key in event.keys():
                standard_attrs = ["concept:name", "time:timestamp", "lifecycle:transition", "org:resource"]
                if attr_key not in standard_attrs:
                    data_attributes.add(attr_key)
    
    # Step 2: Add data nodes to DDG
    for data_attr in data_attributes:
        ddg.add_node(data_attr, node_type="data")
    
    # Step 3: Add activity nodes to DDG
    for activity in activity_names:
        ddg.add_node(activity, node_type="activity")
    
    # Step 4: Extract input-output relations for each activity (paper Table 3)
    activity_io_map = {}
    for activity in activity_names:
        input_data = set()
        output_data = set()
        
        # Collect input and output data for the activity from event log
        for trace in event_log:
            for event_idx, event in enumerate(trace):
                if event["concept:name"] != activity:
                    continue
                
                # Output data: attributes written by this event
                standard_attrs = ["concept:name", "time:timestamp", "lifecycle:transition", "org:resource"]
                for attr_key, attr_value in event.items():
                    if attr_key not in standard_attrs:
                        output_data.add(attr_key)
                
                # Input data: attributes from previous events used by this activity
                if event_idx > 0:
                    prev_event = trace[event_idx - 1]
                    for attr_key, attr_value in prev_event.items():
                        if attr_key not in standard_attrs and attr_key in event:
                            if event[attr_key] == prev_event[attr_key]:
                                input_data.add(attr_key)
        
        activity_io_map[activity] = {
            "input": input_data,
            "output": output_data
        }
    
    # Step 5: Add edges to DDG based on 6 impact patterns (paper Table 3)
    for activity, io_data in activity_io_map.items():
        input_data = io_data["input"]
        output_data = io_data["output"]
        
        # Pattern 1: Activity affects data (A->D)
        for data_attr in output_data:
            ddg.add_edge(activity, data_attr, edge_type="A->D")
        
        # Pattern 2: Data affects activity (D->A)
        for data_attr in input_data:
            ddg.add_edge(data_attr, activity, edge_type="D->A")
        
        # Pattern 3: Data affects data through activity (D->A->D)
        for in_data in input_data:
            for out_data in output_data:
                ddg.add_edge(in_data, out_data, edge_type="D->A->D")
                ddg.add_edge(in_data, activity, edge_type="D->A")
                ddg.add_edge(activity, out_data, edge_type="A->D")
    
    # Step 6: Add path constraint nodes and edges
    # Extract path constraints from process model (gateway conditions)
    from pm4py.objects.bpmn.obj import BPMN
    gateways = [node for node in process_model.get_nodes() if isinstance(node, BPMN.ExclusiveGateway)]
    
    for gateway in gateways:
        gateway_id = gateway.get_id()
        gateway_name = gateway.get_name() or f"gateway_{gateway_id}"
        
        # Add path constraint node
        ddg.add_node(gateway_name, node_type="path_constraint")
        
        # Pattern 4: Data affects path constraint (D->P)
        # Assume path constraints use data attributes from event log
        for data_attr in data_attributes:
            if data_attr in gateway_name.lower():
                ddg.add_edge(data_attr, gateway_name, edge_type="D->P")
        
        # Pattern 5: Path constraint affects activity (P->A)
        # Get outgoing sequence flows from gateway
        for out_flow in gateway.get_outgoing():
            target_node = out_flow.get_target()
            if isinstance(target_node, BPMN.Task):
                activity_name = target_node.get_name()
                if activity_name in activity_names:
                    ddg.add_edge(gateway_name, activity_name, edge_type="P->A")
    
    return ddg"""
src/ccpa_algorithm.py
Context-Aware Change Propagation Algorithm (CCPA)
Corresponds to paper Section 5.1 Algorithm 1
"""
from collections import deque
from src.data_dependency_graph import DataDependencyGraph
from pm4py.objects.log.obj import Trace
from configs.config import EXPERIMENT_CONFIG

def CCPA(trace: Trace, ddg: DataDependencyGraph, config=EXPERIMENT_CONFIG):
    """
    Context-Aware Change Propagation Algorithm (CCPA), paper Algorithm 1
    Uses BFS with dynamic pruning to identify affected elements by data changes
    Args:
        trace: PM4Py Trace object
        ddg: DataDependencyGraph object
        config: Global experiment config
    Returns:
        affected_set: Set of all affected elements (activities, data, path constraints)
    """
    # Step 1: Initialize affected set and visited set (Algorithm 1 lines 1-3)
    affected_set = set()
    visited_nodes = set()
    
    # Step 2: Detect changed data attributes in the trace
    changed_data = set()
    previous_data_values = {}
    
    for event in trace:
        standard_attrs = ["concept:name", "time:timestamp", "lifecycle:transition", "org:resource"]
        for attr_key, attr_value in event.items():
            if attr_key in standard_attrs or attr_key.startswith("env_"):
                continue
            # Check if data value changed from previous event
            if attr_key in previous_data_values:
                if previous_data_values[attr_key] != attr_value:
                    changed_data.add(attr_key)
            previous_data_values[attr_key] = attr_value
    
    # If no changed data, return empty affected set
    if len(changed_data) == 0:
        return affected_set
    
    # Step 3: Initialize queue with changed data attributes (Algorithm 1 lines 4-7)
    queue = deque()
    for data_attr in changed_data:
        if data_attr in ddg.data_nodes:
            queue.append(data_attr)
            affected_set.add(data_attr)
    
    # Step 4: BFS traversal with dynamic pruning (Algorithm 1 lines 8-26)
    # Get executed activities in the trace for dynamic pruning
    executed_activities = [event["concept:name"] for event in trace]
    
    while queue:
        current_node = queue.popleft()
        
        # Skip if already visited
        if current_node in visited_nodes:
            continue
        visited_nodes.add(current_node)
        
        # Get all successors of current node
        successors = ddg.get_successors(current_node)
        
        for (target_node, edge_type) in successors:
            # Dynamic pruning: skip nodes not related to current trace execution
            node_type = ddg.get_node_type(target_node)
            if node_type == "activity" and target_node not in executed_activities:
                # Activity not executed in this trace, prune this path
                continue
            
            # Add to affected set
            affected_set.add((current_node, target_node, edge_type))
            affected_set.add(target_node)
            
            # Determine whether to continue propagation based on edge type
            if edge_type in ["D->A", "A->D", "D->A->D", "D->P", "P->A"]:
                # Continue propagation for these edge types
                if target_node not in visited_nodes:
                    queue.append(target_node)
            # For other edge types, only record impact, no further propagation
    
    # Step 5: Return affected set (Algorithm 1 line 27)
    return affected_set