set -ex
python main.py darkflpg boosted --ft full --suffix exps/test/$1/cifar/full_boosted_100 --device $2 --dataset cifar100_noniid1000 --model vit --sr 0.1 --total_num 100 --lr 0.05 --bs 32 --noise 100
python main.py darkflpg boosted --ft full --suffix exps/test/$1/cifar/full_boosted_150 --device $2 --dataset cifar100_noniid1000 --model vit --sr 0.1 --total_num 100 --lr 0.05 --bs 32 --noise 150
python main.py darkflpg boosted --ft full --suffix exps/test/$1/cifar/full_boosted_200 --device $2 --dataset cifar100_noniid1000 --model vit --sr 0.1 --total_num 100 --lr 0.05 --bs 32 --noise 200
python main.py darkflpg boosted --ft full --suffix exps/test/$1/cifar/full_boosted_500 --device $2 --dataset cifar100_noniid1000 --model vit --sr 0.1 --total_num 100 --lr 0.05 --bs 32 --noise 500
