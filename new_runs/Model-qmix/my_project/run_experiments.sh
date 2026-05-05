#!/bin/bash
echo "========================================================="
echo "Running Full 4-Method Comparison (3000 Episodes)"
echo "========================================================="
echo ""

COMMON_ARGS="--total_episodes 3000 --anneal_steps 2000 --model_learning_period 1000 --alg qmix --map RDM --n_agents 3 --load_model True"

echo "--> Starting: Priority Aware + Deadline + Load Balanced"
python main.py $COMMON_ARGS --model True --priority --federated True --tag _priority_final
echo ""

echo "--> Starting: Baseline FedQMIX"
python main.py $COMMON_ARGS --model True --federated True --tag _baseline_fedqmix
echo ""

echo "--> Starting: Model-Aided QMIX (Centralized QMIX with Channel Learning)"
python main.py $COMMON_ARGS --model True --federated False --tag _model_qmix
echo ""

echo "--> Starting: QMIX (Pure Centralized QMIX)"
python main.py $COMMON_ARGS --model False --federated False --tag _pure_qmix
echo ""

echo "All experiments completed."
