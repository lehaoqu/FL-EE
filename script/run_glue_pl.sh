set -ex

md=bert
cp=models/google-bert/bert-12-uncased

sr=0.1
total_num=120


bs=32
optim=sgd

policies=(base boosted l2w)
datasets=(sst2)

# policies=(base boosted)
# datasets=(mrpc sst2 qnli qqp)
ply=$4

for ds in "${datasets[@]}"
do

    echo "$ply" 
    # python main.py exclusivefl  $ply --suffix $1/$ds --device $2 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $3 --bs $bs --config_path $cp --optim $optim
    # python main.py reefl        $ply --suffix $1/$ds --device $2 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $3 --bs $bs --config_path $cp --optim $optim
    python main.py eefl         $ply --suffix $1/$ds --device $2 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $3 --bs $bs --config_path $cp --optim $optim --epoch 5
    python main.py inclusivefl  $ply --suffix $1/$ds --device $2 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $3 --bs $bs --config_path $cp --optim $optim
    python main.py depthfl      $ply --suffix $1/$ds --device $2 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $3 --bs $bs --config_path $cp --optim $optim
    python main.py scalefl      $ply --suffix $1/$ds --device $2 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $3 --bs $bs --config_path $cp --optim $optim
    # python main.py heterofl     $ply --suffix $1/$ds --device $2 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $3 --bs $bs --config_path $cp --optim $optim

done
