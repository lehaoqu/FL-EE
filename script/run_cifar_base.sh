set -ex

sr=0.1
total_num=100

md=vit
bs=32
lr=0.05

noniids=(1000 1 0.1)

for noniid in "${noniids[@]}"
do
        python main.py inclusivefl    $3  --ft $2 --suffix $1/${2}_${3}/noniid$noniid --device $4 --dataset cifar100_noniid$noniid --model $md --sr $sr --total_num $total_num --lr $4 --bs $bs
        # python main.py eefl             $3  --ft $2 --suffix $1/${2}_${3}/noniid$noniid --device $4 --dataset cifar100_noniid$noniid --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs
        # python main.py depthfl          $3  --ft $2 --suffix $1/${2}_${3}/noniid$noniid --device $4 --dataset cifar100_noniid$noniid --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs
        # python main.py darkflpg         $3  --ft $2 --suffix $1/${2}_${3}/noniid$noniid --device $4 --dataset cifar100_noniid$noniid --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs
        # python main.py scalefl          $3  --ft $2 --suffix $1/${2}_${3}/noniid$noniid --device $4 --dataset cifar100_noniid$noniid --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs
        python main.py reefl            $3  --ft $2 --suffix $1/${2}_${3}/noniid$noniid --device $4 --dataset cifar100_noniid$noniid --model $md --sr $sr --total_num $total_num --lr 0.005 --bs $bs
        # python main.py exclusivefl      $3  --ft $2 --suffix $1/${2}_${3}/noniid$noniid --device $4 --dataset cifar100_noniid$noniid --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs

done
