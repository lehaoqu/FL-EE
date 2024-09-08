set -ex

python main.py ours2 base --suffix exps/tmd --device 0 --dataset cifar100-224-d03 --model vit --sr 0.1 --total_num 120 --lr 0.05 --is_latent True --kd_lr 1e-3 --g_lr 1e-2 --g_eta 0.1 --g_gamma 1 
python main.py ours2 base --suffix exps/tmd --device 0 --dataset cifar100-224-d03 --model vit --sr 0.1 --total_num 120 --lr 0.05 --is_latent True --kd_lr 1e-3 --g_lr 1e-2 --g_eta 0.5 --g_gamma 1
python main.py ours2 base --suffix exps/tmd --device 0 --dataset cifar100-224-d03 --model vit --sr 0.1 --total_num 120 --lr 0.05 --is_latent True --kd_lr 1e-3 --g_lr 1e-2 --g_eta 5   --g_gamma 1

python main.py ours2 base --suffix exps/tmd --device 0 --dataset cifar100-224-d03 --model vit --sr 0.1 --total_num 120 --lr 0.05 --is_latent True --kd_lr 1e-3 --g_lr 1e-2 --g_gamma 1 
python main.py ours2 base --suffix exps/tmd --device 0 --dataset cifar100-224-d03 --model vit --sr 0.1 --total_num 120 --lr 0.05 --is_latent True --kd_lr 1e-3 --g_lr 1e-2 --g_gamma 5
python main.py ours2 base --suffix exps/tmd --device 0 --dataset cifar100-224-d03 --model vit --sr 0.1 --total_num 120 --lr 0.05 --is_latent True --kd_lr 1e-3 --g_lr 1e-2 --g_gamma 10