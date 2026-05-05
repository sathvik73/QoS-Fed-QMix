@echo off
echo =========================================================
echo Running Full 4-Method Comparison (3000 Episodes)
echo =========================================================
echo.

set COMMON_ARGS=--total_episodes 30000 --anneal_steps 2000 --model_learning_period 1000 --alg qmix --map RDM --n_agents 3 --load_model True

echo --^> Starting: Priority Aware + Deadline + Load Balanced
python main.py %COMMON_ARGS% --model True --priority --federated True --tag _proposedf
echo.

@REM echo --^> Starting: Baseline FedQMIX
@REM python main.py %COMMON_ARGS% --model True --federated True --tag _baseline_fedqmix
@REM echo.

@REM echo --^> Starting: Model-Aided QMIX (Centralized QMIX with Channel Learning)
@REM python main.py %COMMON_ARGS% --model True --federated False --tag _model_qmix
@REM echo.

@REM echo --^> Starting: QMIX (Pure Centralized QMIX)
@REM python main.py %COMMON_ARGS% --model False --federated False --tag _pure_qmix
@REM echo.

echo All experiments completed.
pause
