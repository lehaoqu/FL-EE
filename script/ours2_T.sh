set -ex

python main.py ours2 base --suffix exps/sgd --device $1 --dataset cifar100-224-d03 --model vit --sr 0.1 --total_num 120 --lr 0.05 --is_latent True --kd_lr $2 
python main.py ours2 boosted --suffix exps/sgd --device $1 --dataset cifar100-224-d03 --model vit --sr 0.1 --total_num 120 --lr 0.05 --is_latent True --kd_lr $2 
python main.py ours2 l2w --suffix exps/sgd --device $1 --dataset cifar100-224-d03 --model vit --sr 0.1 --total_num 120 --lr 0.05 --is_latent True --kd_lr $2 