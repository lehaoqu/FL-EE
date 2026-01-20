set -ex

sr=0.1
total_num=100

md=vit
bs=32
lr=0.05

# noniids=(1000 1 0.1)
noniids=(1000)

for noniid in "${noniids[@]}"
do
        python main.py reefl            $3  --ft $2 --suffix $1/${2}_${3}/noniid$noniid --device $4 --dataset cifar100_noniid$noniid --model $md --sr $sr --total_num $total_num --lr 0.005     --bs $bs --slimmable --slim_ratios 1.0 0.95     --slim_ce --slim_kd
        python main.py reefl            $3  --ft $2 --suffix $1/${2}_${3}/noniid$noniid --device $4 --dataset cifar100_noniid$noniid --model $md --sr $sr --total_num $total_num --lr 0.005     --bs $bs --slimmable --slim_ratios 1.0 0.9      --slim_ce --slim_kd
        python main.py reefl            $3  --ft $2 --suffix $1/${2}_${3}/noniid$noniid --device $4 --dataset cifar100_noniid$noniid --model $md --sr $sr --total_num $total_num --lr 0.005     --bs $bs --slimmable --slim_ratios 1.0 0.85     --slim_ce --slim_kd
        python main.py reefl            $3  --ft $2 --suffix $1/${2}_${3}/noniid$noniid --device $4 --dataset cifar100_noniid$noniid --model $md --sr $sr --total_num $total_num --lr 0.005     --bs $bs --slimmable --slim_ratios 1.0 0.8      --slim_ce --slim_kd
        python main.py reefl            $3  --ft $2 --suffix $1/${2}_${3}/noniid$noniid --device $4 --dataset cifar100_noniid$noniid --model $md --sr $sr --total_num $total_num --lr 0.005     --bs $bs --slimmable --slim_ratios 1.0 0.7      --slim_ce --slim_kd
        python main.py reefl            $3  --ft $2 --suffix $1/${2}_${3}/noniid$noniid --device $4 --dataset cifar100_noniid$noniid --model $md --sr $sr --total_num $total_num --lr 0.005     --bs $bs --slimmable --slim_ratios 1.0 0.6      --slim_ce --slim_kd
        python main.py reefl            $3  --ft $2 --suffix $1/${2}_${3}/noniid$noniid --device $4 --dataset cifar100_noniid$noniid --model $md --sr $sr --total_num $total_num --lr 0.005     --bs $bs --slimmable --slim_ratios 1.0 0.95 0.9 --slim_ce --slim_kd
done
