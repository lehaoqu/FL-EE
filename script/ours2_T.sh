set -ex

python main.py ours2 $1 --suffix exps/l2w --device 2 --dataset cifar100-224-d03 --model vit --sr 0.1 --total_num 120 --lr 0.05 --is_latent True --kd_lr 1e-3 --g_lr 1e-2 --g_eta 5 --g_gamma 10
python main.py ours2 $1 --suffix exps/l2w --device 2 --dataset cifar100-224-d03 --model vit --sr 0.1 --total_num 120 --lr 0.05 --is_latent True --kd_lr 1e-3 --g_lr 1e-2 --g_eta 5 --g_gamma 20 
python main.py ours2 $1 --suffix exps/l2w --device 2 --dataset cifar100-224-d03 --model vit --sr 0.1 --total_num 120 --lr 0.05 --is_latent True --kd_lr 1e-3 --g_lr 1e-2 --g_eta 5   --g_gamma 100

python main.py ours2 $1 --suffix exps/l2w --device 2 --dataset cifar100-224-d03 --model vit --sr 0.1 --total_num 120 --lr 0.05 --is_latent True --kd_lr 1e-3 --g_lr 1e-2 --g_eta 1 --g_gamma 10
python main.py ours2 $1 --suffix exps/l2w --device 2 --dataset cifar100-224-d03 --model vit --sr 0.1 --total_num 120 --lr 0.05 --is_latent True --kd_lr 1e-3 --g_lr 1e-2 --g_eta 1 --g_gamma 20 
python main.py ours2 $1 --suffix exps/l2w --device 2 --dataset cifar100-224-d03 --model vit --sr 0.1 --total_num 120 --lr 0.05 --is_latent True --kd_lr 1e-3 --g_lr 1e-2 --g_eta 1   --g_gamma 100

python main.py ours2 $1 --suffix exps/l2w --device 2 --dataset cifar100-224-d03 --model vit --sr 0.1 --total_num 120 --lr 0.05 --is_latent True --kd_lr 1e-3 --g_lr 1e-2 --g_eta 10 --g_gamma 10
python main.py ours2 $1 --suffix exps/l2w --device 2 --dataset cifar100-224-d03 --model vit --sr 0.1 --total_num 120 --lr 0.05 --is_latent True --kd_lr 1e-3 --g_lr 1e-2 --g_eta 10   --g_gamma 20
python main.py ours2 $1 --suffix exps/l2w --device 2 --dataset cifar100-224-d03 --model vit --sr 0.1 --total_num 120 --lr 0.05 --is_latent True --kd_lr 1e-3 --g_lr 1e-2 --g_eta 10   --g_gamma 100
