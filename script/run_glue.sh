set -ex

md=bert
cp=models/bert_uncased_L-12_H-128_A-2

sr=0.1
total_num=120


bs=32
optim=sgd

lr=5e-2
epoch=1


datasets=(qqp qnli)

ply=$3

for ds in "${datasets[@]}"
do


    echo "$ply" 
    if [ "$ply" == "boosted" ]; then
        python main.py scalefl      $ply --suffix $1/$ds --device $2 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --config_path $cp --optim $optim --epoch $epoch
        python main.py exclusivefl  $ply --suffix $1/$ds --device $2 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --config_path $cp --optim $optim --epoch $epoch
        python main.py eefl         $ply --suffix $1/$ds --device $2 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --config_path $cp --optim $optim --epoch $epoch
        # python main.py inclusivefl  $ply --suffix $1/$ds --device $2 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --config_path $cp --optim $optim --epoch $epoch
        python main.py depthfl      $ply --suffix $1/$ds --device $2 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --config_path $cp --optim $optim --epoch $epoch
        # python main.py reefl        $ply --suffix $1/$ds --device $2 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --config_path $cp --optim $optim --epoch $epoch
        
    else
        python main.py scalefl      $ply --suffix $1/$ds --device $2 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --config_path $cp --optim $optim --epoch $epoch
        python main.py exclusivefl  $ply --suffix $1/$ds --device $2 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --config_path $cp --optim $optim --epoch $epoch
        python main.py eefl         $ply --suffix $1/$ds --device $2 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --config_path $cp --optim $optim --epoch $epoch
        # python main.py inclusivefl  $ply --suffix $1/$ds --device $2 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --config_path $cp --optim $optim --epoch $epoch
        python main.py depthfl      $ply --suffix $1/$ds --device $2 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --config_path $cp --optim $optim --epoch $epoch
        python main.py reefl        $ply --suffix $1/$ds --device $2 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --config_path $cp --optim $optim --epoch $epoch
        
    fi


done
