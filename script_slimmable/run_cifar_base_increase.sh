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


        python main.py eefl             $3  --ft $2 --suffix $1/${2}_${3}/noniid$noniid --device $4 --dataset cifar100_noniid$noniid --model $md --sr $sr --total_num $total_num --lr $lr       --bs $bs --slimmable --slim_ratios 1.0 0.25
        python main.py eefl             $3  --ft $2 --suffix $1/${2}_${3}/noniid$noniid --device $4 --dataset cifar100_noniid$noniid --model $md --sr $sr --total_num $total_num --lr $lr       --bs $bs --slimmable --slim_ratios 1.0 0.5 0.25
        python main.py eefl             $3  --ft $2 --suffix $1/${2}_${3}/noniid$noniid --device $4 --dataset cifar100_noniid$noniid --model $md --sr $sr --total_num $total_num --lr $lr       --bs $bs --slimmable --slim_ratios 1.0 0.75 0.5 0.25

done
