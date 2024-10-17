set -ex

sr=0.1
total_num=120

# dts=cifar100-224-d03
# dts=cifar100-224-d03-1
md=vit
bs=32


ply=$4

for ply in "${policies[@]}"
do
    echo "$ply" 
    # python main.py ours2        $ply --suffix $1 --device $2 --dataset $3 --model $md --sr $sr --total_num $total_num --lr $4 --bs $bs --is_latent True --kd_lr 1e-3 --g_lr 1e-2 --g_eta 5 --g_gamma 10 
    # python main.py inclusivefl  $ply --suffix $1 --device $2 --dataset $3 --model $md --sr $sr --total_num $total_num --lr $4 --bs $bs
    python main.py exclusivefl  $4 --suffix $1 --device $2 --dataset $3 --model $md --sr $sr --total_num $total_num --lr $4 --bs $bs
    python main.py depthfl      $4 --suffix $1 --device $2 --dataset $3 --model $md --sr $sr --total_num $total_num --lr $4 --bs $bs
    python main.py scalefl      $4 --suffix $1 --device $2 --dataset $3 --model $md --sr $sr --total_num $total_num --lr $4 --bs $bs
    python main.py eefl         $4 --suffix $1 --device $2 --dataset $3 --model $md --sr $sr --total_num $total_num --lr $4 --bs $bs
    python main.py reefl        $4 --suffix $1 --device $2 --dataset $3 --model $md --sr $sr --total_num $total_num --lr $4 --bs $bs
    
done



