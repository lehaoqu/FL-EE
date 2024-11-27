set -ex

sr=0.1
total_num=100

bs=32
lr=0.05
md=bert
cp=models/google-bert/bert-12-128-uncased

datasets=(sst2 qnli qqp)

for ds in "${datasets[@]}"
do
    # python main.py eefl             $3  --rnd 300 --gamma 0.998 --ft $2 --suffix $1/${2}_${3}/$ds --device $4 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $lr    --bs $bs --config_path $cp
    python main.py darkflpg         $3  --rnd 300 --gamma 0.998 --ft $2 --suffix $1/${2}_${3}/$ds --device $4 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $lr    --bs $bs --config_path $cp 
    # python main.py depthfl          $3  --rnd 300 --gamma 0.998 --ft $2 --suffix $1/${2}_${3}/$ds --device $4 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $lr    --bs $bs --config_path $cp
    # python main.py inclusivefl      $3  --rnd 300 --gamma 0.998 --ft $2 --suffix $1/${2}_${3}/$ds --device $4 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $lr    --bs $bs --config_path $cp
    # python main.py scalefl          $3  --rnd 300 --gamma 0.998 --ft $2 --suffix $1/${2}_${3}/$ds --device $4 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $lr    --bs $bs --config_path $cp
    # python main.py reefl            $3  --rnd 300 --gamma 0.998 --ft $2 --suffix $1/${2}_${3}/$ds --device $4 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $lr    --bs $bs --config_path $cp
    # python main.py exclusivefl      $3  --rnd 300 --gamma 0.998 --ft $2 --suffix $1/${2}_${3}/$ds --device $4 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $lr    --bs $bs --config_path $cp
done
