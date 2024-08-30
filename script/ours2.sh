set -ex

python main.py ours2 boosted --suffix exps/tmp --device 3 --dataset cifar100-224-d03 --model vit --sr 0.1 --total_num 120 --lr 0.05
python main.py ours2 l2w --suffix exps/tmp --device 3 --dataset cifar100-224-d03 --model vit --sr 0.1 --total_num 120 --lr 0.05