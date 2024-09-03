set -ex

python main.py ours1 base --suffix exps/cifar_0.05 --device 1 --dataset cifar100-224-d03 --model vit --sr 0.1 --total_num 120 --lr 0.05 --is_latent True
python main.py ours1 boosted --suffix exps/cifar_0.05 --device 1 --dataset cifar100-224-d03 --model vit --sr 0.1 --total_num 120 --lr 0.05 --is_latent True
python main.py ours1 l2w --suffix exps/cifar_0.05 --device 1 --dataset cifar100-224-d03 --model vit --sr 0.1 --total_num 120 --lr 0.05 --is_latent True