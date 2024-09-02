set -ex

sr=0.1
total_num=120

dts=cifar100-224-d03
md=vit
bs=32

policies=(base boosted l2w)

for ply in "${policies[@]}"
do
    echo "$ply" 
    python main.py heterofl $ply --suffix $1 --device $2 --dataset $dts --model $md --sr $sr --total_num $total_num --lr $3 --bs $bs
    python main.py reefl $ply --suffix $1 --device $2 --dataset $dts --model $md --sr $sr --total_num $total_num --lr $3 --bs $bs
    python main.py depthfl $ply --suffix $1 --device $2 --dataset $dts --model $md --sr $sr --total_num $total_num --lr $3 --bs $bs
    python main.py scalefl $ply --suffix $1 --device $2 --dataset $dts --model $md --sr $sr --total_num $total_num --lr $3 --bs $bs
    python main.py inclusivefl $ply --suffix $1 --device $2 --dataset $dts --model $md --sr $sr --total_num $total_num --lr $3 --bs $bs
    python main.py exclusivefl $ply --suffix $1 --device $2 --dataset $dts --model $md --sr $sr --total_num $total_num --lr $3 --bs $bs
done



