set -ex

sr=0.1
total_num=100

bs=32
lr=0.05
md=bert
cp=models/google-bert/bert-12-128-uncased
g=0.99
datasets=(sst2 qnli qqp)

for ds in "${datasets[@]}"
do

    # python main.py darkflpg         $3  --ft $2 --suffix $1/${2}_${3}/$ds --device $4 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $lr    --bs $bs --config_path $cp --noise 10 --s_epoches 2
    python main.py darkflpg         $3 --s_gamma 0.99 --ft $2 --suffix $1/${2}_${3}/$ds --device $4 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $lr    --bs $bs --config_path $cp --noise 2 --s_epoches 2
    # python main.py darkflpa2        $3    --ft $2 --suffix $1/${2}_${3}/$ds --device $4 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $lr    --bs $bs --config_path $cp --noise 2 
    
    # python main.py eefl             $3    --ft $2 --suffix $1/${2}_${3}/$ds --device $4 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $lr    --bs $bs --config_path $cp
    # python main.py depthfl          $3    --ft $2 --suffix $1/${2}_${3}/$ds --device $4 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $lr    --bs $bs --config_path $cp
    # python main.py reefl            $3    --ft $2 --suffix $1/${2}_${3}/$ds --device $4 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $lr    --bs $bs --config_path $cp
    # python main.py inclusivefl      $3    --ft $2 --suffix $1/${2}_${3}/$ds --device $4 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $lr    --bs $bs --config_path $cp
    # python main.py scalefl          $3    --ft $2 --suffix $1/${2}_${3}/$ds --device $4 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $lr    --bs $bs --config_path $cp
    # python main.py exclusivefl      $3    --ft $2 --suffix $1/${2}_${3}/$ds --device $4 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $lr    --bs $bs --config_path $cp
done
