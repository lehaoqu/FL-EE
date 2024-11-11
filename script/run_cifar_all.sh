set -ex

sr=0.1
total_num=120

dts=cifar100-224-d03
# dts=cifar100-224-d03-1
md=vit
bs=32
lr=0.05
ply=$3

echo "$ply" 
python global.py    depthfl     $3 --suffix $1 --device $2 --dataset $dts --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs

suffix=$1
noniid=$2
algs=(eefl depthfl darkflpg)
policies=(boosted base)

for alg in "${algs[@]}"
do
    for ply in "${policies[@]}"
    do
        python main.py $alg $ply    --suffix $suffix/noniid$2/normal --device $2 --dataset cifar100_noniid$2 --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --eq_ratios 0.25  0.25  0.25  0.25
        python main.py $alg $ply    --suffix $suffix/noniid$2/small  --device $2 --dataset cifar100_noniid$2 --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --eq_ratios 0.75  0.083 0.083 0.083
        python main.py $alg $ply    --suffix $suffix/noniid$2/large  --device $2 --dataset cifar100_noniid$2 --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --eq_ratios 0.083 0.083 0.083 0.75
    done
done