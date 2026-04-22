The system integrates GNN-based topology-aware embeddings with reinforcement learning to achieve adaptive and scalable resource allocation in dynamic IoT networks.

EXECUTION STEPS


Step 1: Install dependencies

pip install tensorflow networkx numpy

Step 2: Run Training

python 6gnn.py

Output:

HIGH_best.weights.h5

LOW_best.weights.h5

MULTI_best.weights.h5

Step 3: Run Testing

python gnn_testing.py
 
Step 4: Output Example

Acceptance Ratio: 82.30%

Revenue/Cost Ratio: 1.33

CPU Utilization: 16.6%


 COMPLETE FLOW SUMMARY
1. Generate IoT graph
2. Generate services
3. Classify services
4. Extract features
5. Apply GNN → embeddings
6. Policy network → decisions
7. Allocate nodes + links
8. Compute reward
9. Train model
10. Save weights
11. Load model in testing
12. Evaluate metrics

My work

 Uses GNN for topology awareness
 Uses DRL for intelligent decision making
 Handles dynamic environment
 Supports node failure
 Scales efficiently
 Produces realistic evaluation metrics
