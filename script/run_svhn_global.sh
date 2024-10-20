set -ex

sr=0.1
total_num=120

dts=svhn
# dts=cifar100-224-d03-1
md=vit
bs=32
lr=1e-3
ply=$3

echo "$ply" 
python global.py    depthfl     $3 --suffix $1 --device $2 --dataset $dts --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs
