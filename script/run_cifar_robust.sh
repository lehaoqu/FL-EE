set -ex

sr=0.1
total_num=100

md=vit
bs=32
lr=0.05

noniids=(1000 1 0.1)

if [ "$2" == "lora" ]; then
        $=0.005
fi

if [ "$3" == "small" ]; then
        $eq_ratios=( 0.4 0.3 0.2 0.1 )
fi

if [ "$3" == "normal" ]; then
        $eq_ratios=( 0.25 0.25 0.25 0.25 )
fi

if [ "$3" == "large" ]; then
        $eq_ratios=( 0.1 0.2 0.3 0.4 )
fi


for noniid in "${noniids[@]}"
do
        python main.py inclusivefl      $3  --ft $2 --suffix $1/${2}_${3}/noniid$noniid --device $4 --dataset cifar100_noniid$noniid --model $md --sr $sr --total_num $total_num --lr $lr       --bs $bs --eq_ratios $eq_ratios
        python main.py depthfl          $3  --ft $2 --suffix $1/${2}_${3}/noniid$noniid --device $4 --dataset cifar100_noniid$noniid --model $md --sr $sr --total_num $total_num --lr $lr       --bs $bs --eq_ratios $eq_ratios
        python main.py darkflpg         $3  --ft $2 --suffix $1/${2}_${3}/noniid$noniid --device $4 --dataset cifar100_noniid$noniid --model $md --sr $sr --total_num $total_num --lr $lr       --bs $bs --eq_ratios $eq_ratios
        python main.py scalefl          $3  --ft $2 --suffix $1/${2}_${3}/noniid$noniid --device $4 --dataset cifar100_noniid$noniid --model $md --sr $sr --total_num $total_num --lr $lr       --bs $bs --eq_ratios $eq_ratios
        python main.py reefl            $3  --ft $2 --suffix $1/${2}_${3}/noniid$noniid --device $4 --dataset cifar100_noniid$noniid --model $md --sr $sr --total_num $total_num --lr 0.005     --bs $bs --eq_ratios $eq_ratios
done
