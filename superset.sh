# Basic
#python ./main.py -lvl b -m train -t PPO_Standard
#python ./main.py -lvl b -m train -t PPO_RewardShaping
#python ./main.py -lvl b -m train -t PPO_Curriculum
python ./main.py -lvl b -m train -t PPO_RewardShaping_and_Curriculum
# Defend the Center
#python ./main.py -lvl dtc -m train -t PPO_Standard
#python ./main.py -lvl dtc -m train -t PPO_RewardShaping
#python ./main.py -lvl dtc -m train -t PPO_Curriculum
#python ./main.py -lvl dtc -m train -t PPO_RewardShaping_and_Curriculum
# Deadly Corridor
#python ./main.py -lvl dc -m train -t PPO_Standard
python ./main.py -lvl dc -m train -t PPO_RewardShaping
#python ./main.py -lvl dc -m train -t PPO_Curriculum
#python ./main.py -lvl dc -m train -t PPO_RewardShaping_and_Curriculum
# Defend the Line
#python ./main.py -lvl dtl -m train -t PPO_Standard
#python ./main.py -lvl dtl -m train -t PPO_RewardShaping
#python ./main.py -lvl dtl -m train -t PPO_Curriculum
#python ./main.py -lvl dtl -m train -t PPO_RewardShaping_and_Curriculum
# Deathmatch
python ./main.py -lvl dm -m train -t PPO_Standard
#python ./main.py -lvl dm -m train -t PPO_RewardShaping
python ./main.py -lvl dm -m train -t PPO_Curriculum
#python ./main.py -lvl dm -m train -t PPO_RewardShaping_and_Curriculum
