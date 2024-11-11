set -ex

sr=0.1
total_num=120

md=vit
bs=32
lr=0.05
suffix=$1
noniid=$2
algs=(eefl depthfl darkflpg)
policies=(boosted base)

for alg in "${algs[@]}"
do
    for ply in "${policies[@]}"
    do
        python main.py $alg $ply    --suffix $suffix/noniid$noniid/normal --device $3 --dataset cifar100_noniid$noniid --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --eq_ratios 0.25  0.25  0.25  0.25
        python main.py $alg $ply    --suffix $suffix/noniid$noniid/small  --device $3 --dataset cifar100_noniid$noniid --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --eq_ratios 0.75  0.083 0.083 0.083
        python main.py $alg $ply    --suffix $suffix/noniid$noniid/large  --device $3 --dataset cifar100_noniid$noniid --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --eq_ratios 0.083 0.083 0.083 0.75
    done
done