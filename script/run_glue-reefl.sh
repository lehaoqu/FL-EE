set -ex

md=bert
cp=models/google-bert/bert-12-uncased

sr=0.1
total_num=120


bs=32
optim=sgd
lr=5e-3



datasets=(sst2 qqp qnli)

for ds in "${datasets[@]}"
do
    # python main.py depthfl base --suffix exps/GLUE/qnli --device $1 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --config_path $cp --optim $optim
    python main.py reefl base   --suffix exps/GLUE/$ds --device $1 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --config_path $cp --optim $optim
    python main.py reefl l2w    --suffix exps/GLUE/$ds --device $1 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --config_path $cp --optim $optim
done      