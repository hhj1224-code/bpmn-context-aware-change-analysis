Python 3.9.0 (tags/v3.9.0:9cf6752, Oct  5 2020, 15:34:40) [MSC v.1927 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>># Corresponding Section: 5.1 Algorithm 1 Context-Aware Change Propagation Algorithm (CCPA)
import pandas as pd
from typing import List, Set, Tuple
from src.data_dependency_graph import ProcessDataDependencyGraph

def context_aware_change_propagation(
    ddg: ProcessDataDependencyGraph,
    trace_df: pd.DataFrame,
    changed_data: List[str],
    injection_event_idx: int
) -> Tuple[Set[str], Set[Tuple[str, str, str]]]:
    """
    Context-Aware Change Propagation Algorithm (CCPA)
    Uses BFS traversal with dynamic pruning to identify process elements affected by data changes.
    Only propagates changes to elements executed after the injection point.

    Parameters
    ----------
    ddg : ProcessDataDependencyGraph
        Pre-constructed Process Data Dependency Graph (DDG)
    trace_df : pd.DataFrame
        DataFrame of a single process trace (case)
    changed_data : List[str]
        List of data attributes that have changed
    injection_event_idx : int
        Index of the event where the change was injected

    Returns
    -------
    Tuple[Set[str], Set[Tuple[str, str, str]]]
        1. affected_set: Set of all process elements affected by the change
        2. impact_edges: Set of edges representing the change propagation path
    """
    # Step 1: Initialization
    affected_set: Set[str] = set()
    visited: Set[str] = set()
    queue: List[str] = []
    impact_edges: Set[Tuple[str, str, str]] = set()

    # Extract activities executed after the injection point for dynamic pruning
    post_injection_activities = set(trace_df.iloc[injection_event_idx:]["activity"].unique())
    data_columns = [
        col for col in trace_df.columns
        if col not in ["case_id", "activity", "timestamp", "resource"]
    ]
    post_injection_data = set(data_columns)

    # Initialize queue with all changed data nodes
    for data in changed_data:
        if data in ddg.G.nodes:
            queue.append(data)
            affected_set.add(data)

    # Step 2: BFS Traversal and Change Propagation
    while queue:
        current_node = queue.pop(0)
        if current_node in visited:
            continue
        visited.add(current_node)

        # Iterate over all outgoing edges of the current node
        for successor in ddg.G.successors(current_node):
            edge_type = ddg.edge_types[(current_node, successor)]
            successor_type = ddg.node_types[successor]

            # Dynamic Pruning Logic: only propagate to relevant elements
            is_relevant = False
            if successor_type == "activity" and successor in post_injection_activities:
                is_relevant = True
            elif successor_type == "data" and successor in post_injection_data:
                is_relevant = True
            elif successor_type == "constraint":
                constraint_targets = [
                    e[1] for e in ddg.G.out_edges(successor)
                    if ddg.node_types[e[1]] == "activity"
                ]
                if len(set(constraint_targets) & post_injection_activities) > 0:
                    is_relevant = True

            # Prune the path if the successor is not relevant
            if not is_relevant:
                continue

            # Record the affected element and propagation edge
            affected_set.add(successor)
            impact_edges.add((current_node, successor, edge_type))

            # Propagation Rules: decide whether to continue propagation
            if edge_type in [
                "data_to_data",
                "activity_to_data",
                "data_to_activity",
                "data_to_constraint"
            ]:
                if successor not in visited:
                    queue.append(successor)
            elif edge_type == "constraint_to_activity":
                continue
            elif edge_type == "data_through_activity_to_data":
                if successor not in trace_df.iloc[:injection_event_idx].columns:
                    if successor not in visited:
                        queue.append(successor)

    # Step 3: Return Results
    return affected_set, impact_edges
