mpiexec -n  8  python3   test/baselines/ddpg/main.py  --seed 1 --env-id CellRobotRLEnv-v0 --batch-size 512 \
    --actor-lr 0.0001 --critic-lr 0.001 --nb-epochs 10 --nb-epoch-cycles 10 --nb-train-steps 1000 --action-dim 2 --noise-type adaptive-param_0.2

mpiexec -n  8  python3   test/baselines/ddpg/main.py  --seed 1 --env-id CellRobotRLEnv-v0 --batch-size 512 \
    --actor-lr 0.0001 --critic-lr 0.001 --nb-epochs 10 --nb-epoch-cycles 10 --nb-train-steps 1000 --action-dim 2 --noise-type normal_0.2


mpiexec -n  8  python3   test/baselines/ddpg/main.py  --seed 1 --env-id CellRobotRLEnv-v0 --batch-size 512 \
    --actor-lr 0.0001 --critic-lr 0.001 --nb-epochs 10 --nb-epoch-cycles 100 --nb-train-steps 100 --action-dim 2 --noise-type adaptive-param_0.2

mpiexec -n  8  python3   test/baselines/ddpg/main.py  --seed 1 --env-id CellRobotRLEnv-v0 --batch-size 512 \
    --actor-lr 0.0001 --critic-lr 0.001 --nb-epochs 50 --nb-epoch-cycles 10  --nb-train-steps 100 --action-dim 2 --noise-type adaptive-param_0.2


