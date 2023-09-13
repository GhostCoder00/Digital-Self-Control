# bash code for reproducing our experiments
project=colorado # colorado, korea, engagenet, daisee
for seed in 0 1 2 3 4
do
    python main_FL.py -p $project -seed $seed -fl FedAvg
    python main_FL.py -p $project -seed $seed -fl FedAdam
    python main_FL.py -p $project -seed $seed -fl FedProx
    python main_FL.py -p $project -seed $seed -fl MOON
    python main_FL.py -p $project -seed $seed -fl FedAwS
    python main_FL.py -p $project -seed $seed -fl TurboSVM
done