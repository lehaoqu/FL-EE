set -ex

sr=0.1
total_num=120

# dts=cifar100-224-d03
# dts=cifar100-224-d03-1
md=vit
bs=32


policies=(base boosted l2w)

for ply in "${policies[@]}"
do
    echo "$ply" 
    # python main.py ours2        $ply --suffix $1 --device $2 --dataset $3 --model $md --sr $sr --total_num $total_num --lr $4 --bs $bs --is_latent True --kd_lr 1e-3 --g_lr 1e-2 --g_eta 5 --g_gamma 10 
    # python main.py inclusivefl  $ply --suffix $1 --device $2 --dataset $3 --model $md --sr $sr --total_num $total_num --lr $4 --bs $bs
    python main.py exclusivefl  $ply --suffix $1 --device $2 --dataset $3 --model $md --sr $sr --total_num $total_num --lr $4 --bs $bs
    python main.py depthfl      $ply --suffix $1 --device $2 --dataset $3 --model $md --sr $sr --total_num $total_num --lr $4 --bs $bs
    python main.py scalefl      $ply --suffix $1 --device $2 --dataset $3 --model $md --sr $sr --total_num $total_num --lr $4 --bs $bs
    python main.py eefl         $ply --suffix $1 --device $2 --dataset $3 --model $md --sr $sr --total_num $total_num --lr $4 --bs $bs
    python main.py reefl        $ply --suffix $1 --device $2 --dataset $3 --model $md --sr $sr --total_num $total_num --lr $4 --bs $bs
    
done



