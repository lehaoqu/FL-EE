set -ex

md=bert
cp=models/bert_uncased_L-12_H-128_A-2

sr=0.1
total_num=120


bs=32
optim=sgd
lr=1e-3
epoch=1


# datasets=(sst2 qnli qqp)

# for ds in "${datasets[@]}"
# do
    
#     echo "$ply"
#     python global.py depthfl      $3 --suffix $1/$ds --device $2 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --config_path $cp --optim $optim --epoch $epoch

# done

python global.py depthfl      $3 --suffix $1/$4 --device $2 --dataset $4 --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --config_path $cp --optim $optim --epoch $epoch
