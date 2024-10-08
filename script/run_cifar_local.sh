set -ex

sr=0.3
total_num=40

# dts=cifar100-224-d03
# dts=cifar100-224-d03-1
md=vit
bs=32


policies=(base boosted l2w)

for ply in "${policies[@]}"
do
    echo "$ply" 
    python main.py exclusivefl  $ply --suffix $1 --device $2 --dataset $3 --model $md --sr $sr --total_num $total_num --lr $4 --bs $bs
done



