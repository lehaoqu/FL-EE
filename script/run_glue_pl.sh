set -ex

md=bert
cp=models/bert_uncased_L-12_H-128_A-2

sr=0.1
total_num=120


bs=32
optim=sgd
lr=5e-2


# datasets=(sst2 qnli qqp)
datasets=(sst2 qnli qqp)
ply=$3

for ds in "${datasets[@]}"
do

    echo "$ply" 
    # python main.py exclusivefl  $ply --suffix $1/$ds --device $2 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --config_path $cp --optim $optim
    # python main.py largefl      $ply --suffix $1/$ds --device $2 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --config_path $cp --optim $optim --eq_ratios 0 0 0 1
    python main.py eefl         $ply --suffix $1/$ds --device $2 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --config_path $cp --optim $optim --eq_ratios 0.75 0.083 0.083 0.083
    # python main.py reefl        $ply --suffix $1/$ds --device $2 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --config_path $cp --optim $optim
    python main.py depthfl      $ply --suffix $1/$ds --device $2 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --config_path $cp --optim $optim --eq_ratios 0.75 0.083 0.083 0.083
    # python main.py scalefl      $ply --suffix $1/$ds --device $2 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --config_path $cp --optim $optim
    # python main.py heterofl     $ply --suffix $1/$ds --device $2 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --config_path $cp --optim $optim


done
