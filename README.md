# bpmn-context-aware-change-analysis
Context-Aware Change Impact Analysis for Integrated Processes- Decisions Models in Ubiquitous Environments - Open Source Code
## Core Contribution
1. A three-layer (Process-Service-Decision) architecture for unified modeling of internal process context and external environmental context
2. Context-Aware Change Propagation Algorithm (CCPA) based on Data Dependency Graph (DDG) for internal data change impact analysis
3. Incremental decision re-evaluation algorithm for environmental change driven decision updating
4. SAC (Service Adherence Criterion) based consistency analysis framework for process-decision integrated models
5. Full experimental validation on 3 public real-world datasets and 1 synthetic dataset, with 4 baseline methods for comparison

## Quick Start
1. Configure the environment: `pip install -r requirements.txt`
2. Download datasets: `python data/download_datasets.py`
3. Reproduce all experiments: `python run_experiment.py`

## Repository Structure
